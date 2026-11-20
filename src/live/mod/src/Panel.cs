using System;
using Godot;

namespace AgentBridge;

/// <summary>
/// Small overlay in the top-right corner, below the game's top bar (plain Godot controls, no custom node classes:
/// mod assemblies are not registered as Godot script assemblies, so everything is wired with C# signal events).
/// With agent mode off only the toggle is shown.
/// </summary>
internal sealed class Panel
{
    private CanvasLayer _layer = null!;
    private PanelContainer _box = null!;
    private CheckButton _toggle = null!;
    private Button _fold = null!;
    private VBoxContainer _details = null!;
    private OptionButton _asc = null!;
    private LineEdit _seed = null!;
    private Button _startBtn = null!;
    private Button _pauseBtn = null!;
    private SpinBox _delay = null!;
    private Label _status = null!;
    private Label _info = null!;
    private bool _folded;

    private const int FontSize = 13;

    public static Panel Create(Node root)
    {
        var p = new Panel();
        p.Build(root);
        return p;
    }

    private static T Small<T>(T c) where T : Control
    {
        c.AddThemeFontSizeOverride("font_size", FontSize);
        return c;
    }

    private void Build(Node root)
    {
        _layer = new CanvasLayer { Layer = 120, Name = "AgentBridgePanel" };

        _box = new PanelContainer { Name = "Box", MouseFilter = Control.MouseFilterEnum.Stop };
        var style = new StyleBoxFlat
        {
            BgColor = new Color(0.05f, 0.05f, 0.08f, 0.78f),
            BorderColor = new Color(0.85f, 0.65f, 0.25f, 0.9f),
            ContentMarginLeft = 8, ContentMarginRight = 8, ContentMarginTop = 4, ContentMarginBottom = 6,
        };
        style.SetBorderWidthAll(1);
        style.SetCornerRadiusAll(6);
        _box.AddThemeStyleboxOverride("panel", style);

        var v = new VBoxContainer();
        v.AddThemeConstantOverride("separation", 3);
        _box.AddChild(v);

        // row 1: toggle + fold + short status
        var row1 = new HBoxContainer();
        _toggle = Small(new CheckButton { Text = "Agent mode", FocusMode = Control.FocusModeEnum.None, TooltipText = "AgentBridge: let the AI controller play (127.0.0.1:47213)" });
        _toggle.Toggled += on =>
        {
            Bridge.SetAgentMode(on);
            Refresh();
        };
        _fold = Small(new Button { Text = "-", FocusMode = Control.FocusModeEnum.None, Flat = true, TooltipText = "fold / unfold the panel" });
        _fold.Pressed += () => { _folded = !_folded; Refresh(); };
        row1.AddChild(_toggle);
        row1.AddChild(_fold);
        v.AddChild(row1);

        _details = new VBoxContainer();
        _details.AddThemeConstantOverride("separation", 3);
        v.AddChild(_details);

        // row 2: ascension + seed
        var row2 = new HBoxContainer();
        row2.AddChild(Small(new Label { Text = "Ascension" }));
        _asc = Small(new OptionButton { FocusMode = Control.FocusModeEnum.None });
        for (int i = 0; i <= 10; i++) _asc.AddItem("A" + i, i);
        _asc.Selected = Math.Clamp(Bridge.PanelAscension, 0, 10);
        _asc.ItemSelected += idx => { Bridge.PanelAscension = (int)idx; Bridge.SaveConfig(); };
        row2.AddChild(_asc);
        _seed = Small(new LineEdit { PlaceholderText = "seed (optional)", CustomMinimumSize = new Vector2(120, 0), FocusMode = Control.FocusModeEnum.Click });
        _seed.TextChanged += t => Bridge.PanelSeed = t;
        _seed.TextSubmitted += _ => _seed.ReleaseFocus();
        row2.AddChild(_seed);
        _details.AddChild(row2);

        // row 3: start + pause
        var row3 = new HBoxContainer();
        _startBtn = Small(new Button { Text = "Start agent run", FocusMode = Control.FocusModeEnum.None, TooltipText = "Start a new Ironclad run at the selected ascension (main menu only)" });
        _startBtn.Pressed += () => { Bridge.PanelStartRun(); Refresh(); };
        _pauseBtn = Small(new Button { Text = "Pause (F9)", FocusMode = Control.FocusModeEnum.None });
        _pauseBtn.Pressed += () => { Bridge.TogglePause(); Refresh(); };
        row3.AddChild(_startBtn);
        row3.AddChild(_pauseBtn);
        _details.AddChild(row3);

        // row 4: delay
        var row4 = new HBoxContainer();
        row4.AddChild(Small(new Label { Text = "Action delay (ms)" }));
        _delay = new SpinBox { MinValue = 0, MaxValue = 5000, Step = 50, Value = Bridge.DelayMs, FocusMode = Control.FocusModeEnum.Click };
        _delay.GetLineEdit().AddThemeFontSizeOverride("font_size", FontSize);
        _delay.ValueChanged += val => { Bridge.DelayMs = (int)val; Bridge.SaveConfig(); };
        row4.AddChild(_delay);
        _details.AddChild(row4);

        _status = Small(new Label { Text = "" });
        _details.AddChild(_status);
        _info = Small(new Label { Text = "", AutowrapMode = TextServer.AutowrapMode.WordSmart, CustomMinimumSize = new Vector2(300, 0) });
        _info.AddThemeColorOverride("font_color", new Color(1f, 0.82f, 0.45f));
        _details.AddChild(_info);

        _layer.AddChild(_box);
        root.AddChild(_layer);
        Place();
        root.GetViewport().SizeChanged += Place;
        Refresh();
    }

    /// <summary>Right edge, just below the game's top bar (the top bar's own buttons stay clickable).</summary>
    private void Place()
    {
        try
        {
            Vector2 vp = _layer.GetViewport().GetVisibleRect().Size;
            Vector2 size = _box.GetCombinedMinimumSize();
            _box.Size = size;
            _box.Position = new Vector2(vp.X - size.X - 10, Math.Max(80f, vp.Y * 0.085f));
        }
        catch (Exception e) { U.Warn("panel place: " + e.Message); }
    }

    public void Refresh()
    {
        if (!U.Valid(_box)) return;
        if (_toggle.ButtonPressed != Bridge.AgentMode) _toggle.SetPressedNoSignal(Bridge.AgentMode);
        bool show = Bridge.AgentMode && !_folded;
        _details.Visible = show;
        _fold.Visible = Bridge.AgentMode;
        _fold.Text = _folded ? "+" : "-";
        if (show)
        {
            _pauseBtn.Text = Bridge.Paused ? "Resume (F9)" : "Pause (F9)";
            _startBtn.Disabled = Bridge.CanStartCached() != null || Bridge.Busy;
            _startBtn.TooltipText = Bridge.CanStartCached() ?? "Start a new Ironclad run at the selected ascension";
            if ((int)_delay.Value != Bridge.DelayMs) _delay.SetValueNoSignal(Bridge.DelayMs);
            if (_asc.Selected != Bridge.PanelAscension) _asc.Selected = Bridge.PanelAscension;
            _status.Text = Bridge.StatusLine();
            string info = "";
            if (Bridge.Paused) info += "Paused: " + (Bridge.PauseReason ?? "") + "\n";
            if (Bridge.LastAction.Length > 0) info += "Last: " + Bridge.LastAction + "\n";
            if (Bridge.LastError.Length > 0 && !(Bridge.Paused && Bridge.PauseReason == Bridge.LastError)) info += "Error: " + Bridge.LastError;
            _info.Text = info.TrimEnd();
        }
        Place();
    }
}
