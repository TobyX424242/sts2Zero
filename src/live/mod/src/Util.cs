using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Godot;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace AgentBridge;

/// <summary>Small helpers: logging, reflection on private game fields, node search, JSON.</summary>
internal static class U
{
    // ---------------------------------------------------------------- logging
    public static void Info(string msg)
    {
        try { MegaCrit.Sts2.Core.Logging.Log.Info("[AgentBridge] " + msg); }
        catch { try { GD.Print("[AgentBridge] " + msg); } catch { } }
    }

    public static void Warn(string msg)
    {
        try { MegaCrit.Sts2.Core.Logging.Log.Warn("[AgentBridge] " + msg); }
        catch { try { GD.Print("[AgentBridge] WARN " + msg); } catch { } }
    }

    public static void Error(string msg)
    {
        try { MegaCrit.Sts2.Core.Logging.Log.Error("[AgentBridge] " + msg); }
        catch { try { GD.PrintErr("[AgentBridge] " + msg); } catch { } }
    }

    public static long NowMs => (long)Time.GetTicksMsec();

    // ---------------------------------------------------------------- reflection
    private const BindingFlags AnyInstance = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

    /// <summary>Reads a (possibly private, possibly inherited) field or property; null when missing.</summary>
    public static T? Get<T>(object? obj, string name) where T : class
    {
        object? v = GetRaw(obj, name);
        return v as T;
    }

    public static object? GetRaw(object? obj, string name)
    {
        if (obj == null) return null;
        try
        {
            for (Type? t = obj.GetType(); t != null; t = t.BaseType)
            {
                FieldInfo? f = t.GetField(name, AnyInstance | BindingFlags.DeclaredOnly);
                if (f != null) return f.GetValue(obj);
                PropertyInfo? p = t.GetProperty(name, AnyInstance | BindingFlags.DeclaredOnly);
                if (p != null && p.GetIndexParameters().Length == 0) return p.GetValue(obj);
            }
        }
        catch (Exception e) { Warn($"reflection read {obj.GetType().Name}.{name} failed: {e.Message}"); }
        return null;
    }

    public static bool TryGetStruct<T>(object? obj, string name, out T value) where T : struct
    {
        value = default;
        object? v = GetRaw(obj, name);
        if (v is T t) { value = t; return true; }
        return false;
    }

    /// <summary>Calls a (possibly private, possibly inherited) instance method by name.</summary>
    public static object? Call(object obj, string name, params object?[] args)
    {
        for (Type? t = obj.GetType(); t != null; t = t.BaseType)
        {
            foreach (MethodInfo m in t.GetMethods(AnyInstance | BindingFlags.DeclaredOnly))
            {
                if (m.Name == name && m.GetParameters().Length == args.Length)
                    return m.Invoke(obj, args);
            }
        }
        throw new MissingMethodException(obj.GetType().Name, name);
    }

    // ---------------------------------------------------------------- nodes
    public static bool Valid(GodotObject? o)
    {
        try { return o != null && GodotObject.IsInstanceValid(o); } catch { return false; }
    }

    public static bool Visible(CanvasItem? c)
    {
        try { return Valid(c) && c!.IsInsideTree() && c.IsVisibleInTree(); } catch { return false; }
    }

    public static bool Usable(NClickableControl? c)
    {
        try { return Visible(c) && c!.IsEnabled && !c.IsQueuedForDeletion(); } catch { return false; }
    }

    public static List<T> FindAll<T>(Node? start) where T : Node
    {
        var list = new List<T>();
        if (Valid(start)) FindAllRec(start!, list);
        return list;
    }

    private static void FindAllRec<T>(Node node, List<T> found) where T : Node
    {
        if (!Valid(node)) return;
        if (node is T t) found.Add(t);
        foreach (Node child in node.GetChildren()) FindAllRec(child, found);
    }

    public static T? FindFirst<T>(Node? start) where T : Node
    {
        if (!Valid(start)) return null;
        if (start is T r) return r;
        foreach (Node child in start!.GetChildren())
        {
            T? v = FindFirst<T>(child);
            if (v != null) return v;
        }
        return null;
    }

    /// <summary>True when some ancestor (up to stop) has a name containing the text.</summary>
    public static bool HasAncestorNamed(Node n, string text, Node stop)
    {
        for (Node? p = n.GetParent(); p != null && p != stop; p = p.GetParent())
            if (p.Name.ToString().Contains(text, StringComparison.OrdinalIgnoreCase)) return true;
        return false;
    }

    public static string TypeName(object? o) => o?.GetType().Name ?? "null";

    // ---------------------------------------------------------------- json
    public static readonly JsonSerializerOptions Compact = new() { WriteIndented = false };

    public static JsonArray Arr<T>(IEnumerable<T> items, Func<T, JsonNode?> f)
    {
        var a = new JsonArray();
        foreach (T it in items) a.Add(f(it));
        return a;
    }

    public static string Str(JsonNode? n)
    {
        try { return n?.ToJsonString(Compact) ?? "null"; } catch (Exception e) { return "\"<json error: " + e.Message + ">\""; }
    }

    public static int? IntOf(JsonNode? n)
    {
        try
        {
            if (n is JsonValue v)
            {
                if (v.TryGetValue(out int i)) return i;
                if (v.TryGetValue(out long l)) return (int)l;
                if (v.TryGetValue(out double d)) return (int)d;
                if (v.TryGetValue(out string? s) && int.TryParse(s, out int j)) return j;
            }
        }
        catch { }
        return null;
    }

    public static string? StrOf(JsonNode? n)
    {
        try
        {
            if (n is JsonValue v)
            {
                if (v.TryGetValue(out string? s)) return s;
                return v.ToJsonString();
            }
        }
        catch { }
        return null;
    }

    public static bool? BoolOf(JsonNode? n)
    {
        try { if (n is JsonValue v && v.TryGetValue(out bool b)) return b; } catch { }
        return null;
    }

    /// <summary>Stable 64-bit FNV-1a hash of a string (state fingerprints).</summary>
    public static string Hash(string s)
    {
        ulong h = 14695981039346656037UL;
        foreach (char c in s) { h ^= c; h *= 1099511628211UL; }
        return h.ToString("x16");
    }

    /// <summary>Strip [bbcode] tags from game rich text.</summary>
    public static string Plain(string? text)
    {
        if (string.IsNullOrEmpty(text)) return "";
        var sb = new StringBuilder(text.Length);
        int depth = 0;
        foreach (char c in text)
        {
            if (c == '[') { depth++; continue; }
            if (c == ']' && depth > 0) { depth--; continue; }
            if (depth == 0) sb.Append(c);
        }
        return sb.ToString().Replace('\n', ' ').Trim();
    }

    public static string SafeText(Func<object?> f)
    {
        try
        {
            object? v = f();
            if (v is MegaCrit.Sts2.Core.Localization.LocString ls) return Plain(ls.GetFormattedText());
            return v?.ToString() ?? "";
        }
        catch { return ""; }
    }
}

/// <summary>Stable per-object integer ids (card "instance" field).</summary>
internal static class InstanceIds
{
    private static readonly ConditionalWeakTable<object, object> _ids = new();
    private static int _next = 1;

    public static int Of(object o)
    {
        lock (_ids)
        {
            if (_ids.TryGetValue(o, out object? v)) return (int)v;
            int id = _next++;
            _ids.Add(o, id);
            return id;
        }
    }
}

/// <summary>Append-only JSONL log under user://AgentBridge/logs/ (one file per game session).</summary>
internal static class JsonlLog
{
    private static StreamWriter? _w;
    private static readonly object _lock = new();
    public static string Dir { get; private set; } = "";
    public static string FilePath { get; private set; } = "";

    public static void Init()
    {
        try
        {
            Dir = ProjectSettings.GlobalizePath("user://AgentBridge/logs");
            Directory.CreateDirectory(Dir);
            FilePath = Path.Combine(Dir, $"bridge_{DateTime.Now:yyyyMMdd_HHmmss}.jsonl");
            _w = new StreamWriter(new FileStream(FilePath, FileMode.Append, System.IO.FileAccess.Write, FileShare.Read), new UTF8Encoding(false)) { AutoFlush = true };
            U.Info("JSONL log: " + FilePath);
        }
        catch (Exception e) { U.Error("could not open JSONL log: " + e); }
    }

    public static void Write(string type, JsonNode? payload)
    {
        try
        {
            var o = new JsonObject
            {
                ["t"] = DateTime.UtcNow.ToString("o"),
                ["type"] = type,
                ["data"] = payload?.DeepClone(),
            };
            string line = o.ToJsonString(U.Compact);
            lock (_lock) { _w?.WriteLine(line); }
        }
        catch (Exception e) { U.Warn("jsonl write failed: " + e.Message); }
    }
}
