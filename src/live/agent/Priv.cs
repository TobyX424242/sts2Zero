using System.Reflection;
using System.Text.Json.Nodes;

namespace Sts2Sim.Live;

/// <summary>
/// Reflection access to non-public members of the simulator and of RunAgent. The simulator's InternalsVisibleTo list
/// does not name this assembly, and S3 / the simulator are not edited for live play (shared with the training chain).
/// Every member is looked up by name at use; a missing member throws a MissingMemberException that names it (the
/// simulator or RunAgent changed and this file must follow).
/// </summary>
public static class Priv
{
    private const BindingFlags All = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

    private static FieldInfo Field(Type t, string name)
    {
        for (Type? x = t; x is not null; x = x.BaseType)
            if (x.GetField(name, All | BindingFlags.DeclaredOnly) is { } f) return f;
        throw new MissingMemberException(t.FullName, name);
    }

    private static PropertyInfo Prop(Type t, string name)
    {
        for (Type? x = t; x is not null; x = x.BaseType)
            if (x.GetProperty(name, All | BindingFlags.DeclaredOnly) is { } p) return p;
        throw new MissingMemberException(t.FullName, name);
    }

    public static object? Get(object o, string field) => Field(o.GetType(), field).GetValue(o);
    public static T Get<T>(object o, string field) => (T)Field(o.GetType(), field).GetValue(o)!;
    public static void Set(object o, string field, object? value) => Field(o.GetType(), field).SetValue(o, value);
    public static object? GetStatic(Type t, string field) => Field(t, field).GetValue(null);

    /// <summary>Sets a property through its (possibly private or init-only) setter.</summary>
    public static void SetProp(object o, string prop, object? value)
    {
        PropertyInfo p = Prop(o.GetType(), prop);
        MethodInfo set = p.GetSetMethod(true) ?? throw new MissingMemberException(o.GetType().FullName, prop + ".set");
        set.Invoke(o, new[] { value });
    }

    public static object? GetProp(object o, string prop) => Prop(o.GetType(), prop).GetValue(o);

    private static MethodInfo Method(Type t, string name, object?[] args, bool isStatic)
    {
        for (Type? x = t; x is not null; x = x.BaseType)
        {
            var cands = x.GetMethods(All | BindingFlags.DeclaredOnly)
                .Where(m => m.Name == name && m.IsStatic == isStatic && m.GetParameters().Length == args.Length).ToList();
            foreach (MethodInfo m in cands)
            {
                var ps = m.GetParameters();
                bool ok = true;
                for (int i = 0; i < ps.Length && ok; i++)
                    ok = args[i] is null ? !ps[i].ParameterType.IsValueType || Nullable.GetUnderlyingType(ps[i].ParameterType) is not null
                        : ps[i].ParameterType.IsInstanceOfType(args[i]);
                if (ok) return m;
            }
        }
        throw new MissingMethodException(t.FullName, name + $"/{args.Length}");
    }

    /// <summary>Calls an instance method (any visibility); exceptions of the callee are rethrown unwrapped.</summary>
    public static object? Call(object o, string name, params object?[] args)
    {
        try { return Method(o.GetType(), name, args, false).Invoke(o, args); }
        catch (TargetInvocationException e) when (e.InnerException is not null)
        { System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
    }

    public static object? CallStatic(Type t, string name, params object?[] args)
    {
        try { return Method(t, name, args, true).Invoke(null, args); }
        catch (TargetInvocationException e) when (e.InnerException is not null)
        { System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
    }

    /// <summary>Creates an object through a constructor of any visibility (matched by argument count and types).</summary>
    public static object New(Type t, params object?[] args)
    {
        foreach (ConstructorInfo c in t.GetConstructors(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
        {
            var ps = c.GetParameters();
            if (ps.Length != args.Length) continue;
            bool ok = true;
            for (int i = 0; i < ps.Length && ok; i++)
                ok = args[i] is null ? !ps[i].ParameterType.IsValueType : ps[i].ParameterType.IsInstanceOfType(args[i]);
            if (!ok) continue;
            try { return c.Invoke(args); }
            catch (TargetInvocationException e) when (e.InnerException is not null)
            { System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
        }
        throw new MissingMethodException(t.FullName, $".ctor/{args.Length}");
    }

    public static Type SimType(string fullName) =>
        typeof(Sts2Sim.Core.Runs.RunState).Assembly.GetType(fullName, throwOnError: true)!;
}

/// <summary>Lenient readers of the protocol's JSON (missing or mistyped fields give the default).</summary>
public static class J
{
    public static JsonNode? At(this JsonNode? n, string key) => n is JsonObject o && o.TryGetPropertyValue(key, out JsonNode? v) ? v : null;
    public static JsonObject? Obj(this JsonNode? n, string key) => n.At(key) as JsonObject;
    public static JsonArray Arr(this JsonNode? n, string key) => n.At(key) as JsonArray ?? new JsonArray();
    public static IEnumerable<JsonObject> Objs(this JsonNode? n, string key) => n.Arr(key).OfType<JsonObject>();

    public static string? Str(this JsonNode? n, string key)
    {
        JsonNode? v = n.At(key);
        if (v is JsonValue jv)
        {
            if (jv.TryGetValue(out string? s)) return s;
            if (jv.TryGetValue(out long l)) return l.ToString(System.Globalization.CultureInfo.InvariantCulture);
            if (jv.TryGetValue(out bool b)) return b ? "true" : "false";
        }
        return null;
    }

    public static int Int(this JsonNode? n, string key, int def = 0) => IntOrNull(n, key) ?? def;

    public static int? IntOrNull(this JsonNode? n, string key)
    {
        if (n.At(key) is not JsonValue jv) return null;
        if (jv.TryGetValue(out int i)) return i;
        if (jv.TryGetValue(out long l)) return (int)l;
        if (jv.TryGetValue(out double d)) return (int)Math.Round(d);
        if (jv.TryGetValue(out string? s) && int.TryParse(s, out int k)) return k;
        return null;
    }

    public static double Dbl(this JsonNode? n, string key, double def = 0)
    {
        if (n.At(key) is not JsonValue jv) return def;
        if (jv.TryGetValue(out double d)) return d;
        if (jv.TryGetValue(out long l)) return l;
        return def;
    }

    public static bool Bool(this JsonNode? n, string key, bool def = false)
    {
        if (n.At(key) is not JsonValue jv) return def;
        if (jv.TryGetValue(out bool b)) return b;
        if (jv.TryGetValue(out int i)) return i != 0;
        return def;
    }

    /// <summary>A model id in the simulator's form "CATEGORY.ENTRY" (the game's run-save form; "ENTRY" alone gets
    /// <paramref name="category"/>).</summary>
    public static string ModelId(string? id, string category)
    {
        if (string.IsNullOrEmpty(id)) return "";
        return id.Contains('.') ? id : category + "." + id;
    }

    /// <summary>An enum value written by the game (snake_case "rest_site", PascalCase "RestSite" or an integer).</summary>
    public static bool TryEnum<T>(JsonNode? v, out T value) where T : struct, Enum
    {
        value = default;
        if (v is not JsonValue jv) return false;
        if (jv.TryGetValue(out int i)) { value = (T)Enum.ToObject(typeof(T), i); return true; }
        if (!jv.TryGetValue(out string? s) || s is null) return false;
        return Enum.TryParse(s.Replace("_", ""), ignoreCase: true, out value);
    }
}
