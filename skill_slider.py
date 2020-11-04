import time
from threading import Thread

import wx


class SlidersInputFrame(wx.Frame):
    def __init__(self, dim=3):
        super().__init__(parent=None, title="Sliders", size=(400, dim*100 - 20))
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self._sliders = [_make_slider(panel) for _ in range(dim)]
        for slider in self._sliders:
            sizer.Add(slider, 0, wx.ALL | wx.EXPAND, 5)
        panel.SetSizer(sizer)
        self.Show()

    def get_slider_values(self):
        return [s.GetValue() for s in self._sliders]


def _make_slider(parent):
    style = wx.SL_MIN_MAX_LABELS | wx.SL_VALUE_LABEL
    return wx.Slider(parent, value=0.0, minValue=-3.5, maxValue=3.5, style=style)


def create_sliders_widget(dim=5) -> SlidersInputFrame:
    app = wx.App()
    frame = SlidersInputFrame(dim=dim)
    t = Thread(target=app.MainLoop)
    t.setDaemon(True)
    t.start()
    return frame


if __name__ == '__main__':
    widget = create_sliders_widget()
    while True:
        print(widget.get_slider_values(), flush=True)
        time.sleep(0.05)