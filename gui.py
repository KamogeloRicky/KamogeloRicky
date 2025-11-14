"""
RickyFX Analysis GUI — v5.0 GOAT EDITION (Fully Enhanced)

This version integrates with the Master Orchestrator to provide users with:
- Full five-tiered analysis (Chimera → Liquidity Hunter → Context → GOAT)
- Adaptive Personality Core (market psychology-based persona selection)
- Post-processing (Divergence Detector + Correlation Guard)

Preserves:
- Original analysis rendering (Price, ATR, Lot, News, Liquidity, Verdict, SL, TP, Entry, Time, Confidence, Retest)
- Advanced, Context, Micro, Structure/Liquidity sections
- News headlines list
- Copy Report
- Confirm Trade button (pending orders only; limit/stop logic determined by executor)
- Two-phase lot normalization & volume invalid auto-retry

**v5.0 Update:** GUI now routes all analysis through gui_bridge.py to access the full orchestration.
"""

import os
import sys
import re
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

try:
    from analysis import analyze_pair
except Exception:
    analyze_pair = None

try:
    from gui_bridge import analyze_for_gui
except Exception:
    analyze_for_gui = None
    print("⚠️ GUI Bridge not available. GUI will use direct analysis.py fallback.")

try:
    from trade_executor import place_pending_order
except Exception:
    place_pending_order = None

try:
    from data import list_symbols
except Exception:
    list_symbols = None

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


class Spinner(QtWidgets.QWidget):
    def __init__(self, parent=None, diameter=32, line_width=4, speed_ms=16):
        super().__init__(parent)
        self._angle = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(speed_ms)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFixedSize(diameter, diameter)
        self._line_width = line_width

    def _tick(self):
        self._angle = (self._angle + 8) % 360
        self.update()

    def paintEvent(self, evt):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect().adjusted(2, 2, -2, -2)
        pen = QtGui.QPen(QtGui.QColor("#e6eef8"))
        pen.setWidth(self._line_width)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, int(self._angle * 16), int(270 * 16))
        p.end()


class LoadingOverlay(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setVisible(False)
        self._bg_color = QtGui.QColor(7, 17, 34, 180)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addStretch()
        h = QtWidgets.QHBoxLayout()
        h.addStretch()
        h.addWidget(Spinner(self, diameter=36, line_width=4, speed_ms=16))
        lbl = QtWidgets.QLabel("Analyzing…")
        lbl.setStyleSheet("color:#e6eef8;font-size:14px;font-weight:700;")
        h.addSpacing(8)
        h.addWidget(lbl)
        h.addStretch()
        v.addLayout(h)
        v.addStretch()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), self._bg_color)
        p.end()

    def show_overlay(self):
        if self.parentWidget():
            self.setGeometry(self.parentWidget().rect())
        self.show()

    def hide_overlay(self):
        self.hide()


class Worker(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, pair, timeframe, parent=None):
        super().__init__(parent)
        self.pair = pair
        self.timeframe = timeframe

    def run(self):
        try:
            # NEW: Priority routing to full orchestration via bridge
            if analyze_for_gui is not None:
                print(f"🔬 GUI: Routing {self.pair} analysis to full orchestration...")
                res = analyze_for_gui(self.pair, self.timeframe)
            elif analyze_pair is not None:
                print(f"⚠️ GUI: Bridge unavailable. Using direct GOAT engine fallback.")
                res = analyze_pair(self.pair, self.timeframe)
            else:
                raise RuntimeError("No analysis engine available (neither bridge nor analysis.py found).")
            
            self.result_ready.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class FXGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RickyFX Analysis")
        self.resize(980, 760)
        self.setMinimumSize(880, 680)

        self.worker = None
        self._overlay = None
        self._result_card = None

        self._last_symbol = None
        self._last_verdict = None
        self._last_sl = None
        self._last_tp = None
        self._last_entry = None
        self._last_lot = None

        self.setup_ui()

    def setup_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("RickyFX Analysis v5.0 GOAT")
        title.setStyleSheet("font-size:24px;font-weight:700;color:#ffffff;")
        subtitle = QtWidgets.QLabel("Quantum Entry • Liquidity Hunter • Ultimate Confluence • GOAT Fallback")
        subtitle.setStyleSheet("font-size:12px;color:#d0d4dd;")
        hv = QtWidgets.QVBoxLayout()
        hv.addWidget(title)
        hv.addWidget(subtitle)
        header.addLayout(hv)
        header.addStretch()

        controls_card = QtWidgets.QFrame()
        controls_card.setObjectName("controls_card")
        controls_card.setMinimumHeight(140)
        cv = QtWidgets.QVBoxLayout(controls_card)
        cv.setContentsMargins(12, 12, 12, 12)
        cv.setSpacing(8)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)
        lbl_pair = QtWidgets.QLabel("Forex Pair")
        lbl_pair.setStyleSheet("color:white;font-size:14px;font-weight:600;")
        self.edit_pair = QtWidgets.QLineEdit()
        self.edit_pair.setPlaceholderText("EURUSDc")
        self.edit_pair.setMaximumWidth(220)
        self.edit_pair.setMinimumWidth(140)

        lbl_tf = QtWidgets.QLabel("Timeframe")
        lbl_tf.setStyleSheet("color:white;font-size:14px;font-weight:600;")
        self.combo_tf = QtWidgets.QComboBox()
        self.combo_tf.addItems(["5m", "15m", "30m", "1h"])
        self.combo_tf.setMaximumWidth(120)

        self.btn_analyze = QtWidgets.QPushButton("Analyze")
        self.btn_analyze.setMinimumWidth(140)
        self.btn_analyze.clicked.connect(self.on_analyze_clicked)

        self.btn_confirm = QtWidgets.QPushButton("Confirm Trade")
        self.btn_confirm.setMinimumWidth(140)
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.clicked.connect(self.on_confirm_trade_clicked)

        self.btn_copy = QtWidgets.QPushButton("Copy Report")
        self.btn_copy.setMinimumWidth(120)
        self.btn_copy.setEnabled(False)
        self.btn_copy.clicked.connect(self.copy_report)

        self.btn_list = QtWidgets.QPushButton("List Symbols")
        self.btn_list.setMinimumWidth(120)
        self.btn_list.clicked.connect(self.on_list_symbols)
        if list_symbols is None:
            self.btn_list.setEnabled(False)

        top_row.addWidget(lbl_pair)
        top_row.addWidget(self.edit_pair)
        top_row.addSpacing(6)
        top_row.addWidget(lbl_tf)
        top_row.addWidget(self.combo_tf)
        top_row.addStretch()
        top_row.addWidget(self.btn_analyze)
        top_row.addWidget(self.btn_confirm)
        top_row.addWidget(self.btn_copy)
        top_row.addWidget(self.btn_list)

        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(8)
        lbl_balance = QtWidgets.QLabel("Account Balance")
        lbl_balance.setStyleSheet("color:white;font-size:13px;")
        self.edit_balance = QtWidgets.QLineEdit()
        self.edit_balance.setPlaceholderText("e.g. 1000")
        self.edit_balance.setMaximumWidth(140)

        lbl_risk = QtWidgets.QLabel("Risk %")
        lbl_risk.setStyleSheet("color:white;font-size:13px;")
        self.edit_risk = QtWidgets.QLineEdit()
        self.edit_risk.setPlaceholderText("e.g. 1")
        self.edit_risk.setMaximumWidth(90)

        bottom_row.addWidget(lbl_balance)
        bottom_row.addWidget(self.edit_balance)
        bottom_row.addSpacing(12)
        bottom_row.addWidget(lbl_risk)
        bottom_row.addWidget(self.edit_risk)
        bottom_row.addStretch()

        cv.addLayout(top_row)
        cv.addLayout(bottom_row)

        self.lbl_status = QtWidgets.QLabel("Ready - v5.0 GOAT Edition")
        self.lbl_status.setMinimumHeight(24)
        self.lbl_status.setStyleSheet("font-weight:600;color:#c9d1e3;font-size:13px;")

        result_card = QtWidgets.QFrame()
        self._result_card = result_card
        rv = QtWidgets.QVBoxLayout(result_card)
        rv.setContentsMargins(12, 12, 12, 12)
        rv.setSpacing(6)

        metrics_row = QtWidgets.QHBoxLayout()
        metrics_row.setSpacing(12)
        self.lbl_price = QtWidgets.QLabel("Price: —")
        self.lbl_price.setStyleSheet("font-size:16px;font-weight:700;color:#bfe0a8;")
        self.lbl_atr = QtWidgets.QLabel("ATR: —")
        self.lbl_atr.setStyleSheet("font-size:14px;color:#ffd8b2;")
        self.lbl_lot = QtWidgets.QLabel("Lot: —")
        self.lbl_lot.setStyleSheet("font-size:14px;color:#9ad1ff;font-weight:700;")
        self.lbl_news_status = QtWidgets.QLabel("News: —")
        self.lbl_news_status.setStyleSheet("font-size:13px;color:#d0d4dd;")
        self.lbl_liquidity = QtWidgets.QLabel("Liquidity: —")
        self.lbl_liquidity.setStyleSheet("font-size:13px;color:#d0d4dd;")
        self.lbl_signal = QtWidgets.QLabel("Signal: —")
        self.lbl_signal.setStyleSheet("font-size:16px;font-weight:800;color:#f7c0c0;")

        metrics_row.addWidget(self.lbl_price)
        metrics_row.addSpacing(8)
        metrics_row.addWidget(self.lbl_atr)
        metrics_row.addSpacing(8)
        metrics_row.addWidget(self.lbl_lot)
        metrics_row.addStretch()
        metrics_row.addWidget(self.lbl_news_status)
        metrics_row.addSpacing(10)
        metrics_row.addWidget(self.lbl_liquidity)
        metrics_row.addSpacing(10)
        metrics_row.addWidget(self.lbl_signal)

        self.text_report = QtWidgets.QTextEdit()
        self.text_report.setReadOnly(True)
        self.text_report.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_report.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        self.text_report.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.text_report.setStyleSheet(
            "font-family:'Consolas',monospace;font-size:14px;background:#0f1724;color:#e6eef8;padding:8px;"
        )

        rv.addLayout(metrics_row)
        rv.addWidget(self.text_report)

        label_headlines = QtWidgets.QLabel("News Headlines (sanitized):")
        label_headlines.setStyleSheet("color:#d0d4dd;")
        self.headlines_list = QtWidgets.QListWidget()
        self.headlines_list.setStyleSheet("background:#071122;color:#e6eef8;font-size:13px;")
        self.headlines_list.setMinimumHeight(120)
        self.headlines_list.setWordWrap(True)
        self.headlines_list.setUniformItemSizes(False)
        self.headlines_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        rv.addWidget(label_headlines)
        rv.addWidget(self.headlines_list)

        footer = QtWidgets.QHBoxLayout()
        self.btn_quit = QtWidgets.QPushButton("Quit")
        self.btn_quit.clicked.connect(self.close)
        footer.addStretch()
        footer.addWidget(self.btn_quit)

        root.addLayout(header)
        root.addWidget(controls_card)
        root.addWidget(self.lbl_status)
        root.addWidget(result_card)
        root.addLayout(footer)

        self.setStyleSheet("""
            QWidget { background:#071122; }
            #controls_card { background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0b1724, stop:1 #0f2536); border-radius:12px; }
            QPushButton { background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #2b6ea3, stop:1 #1b486f); color:#ffffff; border-radius:8px; padding:8px 12px; font-weight:700; font-size:14px; }
            QPushButton:disabled { background:#2b3a45; color:#7a8a95; }
            QPushButton:hover { background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #3f8fcf, stop:1 #2a6ba0); }
            QComboBox, QLineEdit { background:#071122; color:#e6eef8; border:1px solid #223343; padding:6px; border-radius:6px; font-size:14px; }
            QTextEdit { border-radius:8px; }
            QListWidget { border-radius:8px; padding:6px; }
        """)

        self._overlay = LoadingOverlay(self._result_card)
        self._result_card.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self._result_card and event.type() == QtCore.QEvent.Resize and self._overlay and self._overlay.isVisible():
            self._overlay.setGeometry(self._result_card.rect())
        return super().eventFilter(watched, event)

    def on_analyze_clicked(self):
        pair = self.edit_pair.text().strip()
        if not pair:
            self.show_error("Please enter a forex pair (e.g. EURUSDc)")
            return
        self._last_symbol = pair
        timeframe = self.combo_tf.currentText()

        bal = self.edit_balance.text().strip()
        risk = self.edit_risk.text().strip()
        if bal:
            try:
                float(bal)
                os.environ["ANALYSIS_ACCOUNT_BALANCE"] = bal
            except Exception:
                pass
        if risk:
            try:
                float(risk)
                os.environ["ANALYSIS_RISK_PCT"] = risk
            except Exception:
                pass

        self.btn_analyze.setEnabled(False)
        self.btn_copy.setEnabled(False)
        self.btn_confirm.setEnabled(False)
        self.lbl_status.setText("Analyzing — running full orchestration...")
        self.text_report.clear()
        self.headlines_list.clear()
        for lbl in (self.lbl_price, self.lbl_atr, self.lbl_signal, self.lbl_news_status, self.lbl_liquidity, self.lbl_lot):
            base = lbl.text().split(":")[0]
            lbl.setText(f"{base}: —")

        if self._overlay:
            self._overlay.show_overlay()

        self.worker = Worker(pair, timeframe)
        self.worker.result_ready.connect(self.on_result_ready)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()

    def on_result_ready(self, res):
        if self._overlay:
            self._overlay.hide_overlay()
        self.btn_analyze.setEnabled(True)
        self.btn_copy.setEnabled(True)

        if not isinstance(res, dict):
            self.show_error("Invalid analysis result")
            return

        def fmt_price(v):
            try:
                return f"{float(v):.6f}"
            except Exception:
                return str(v) if v is not None else "—"

        def fmt_any(v):
            return "—" if v is None else str(v)

        def esc(s):
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        price = res.get("price")
        atr = res.get("atr")
        verdict = res.get("final_suggestion") or res.get("final") or res.get("suggestion")

        sl_val = res.get("sl") or res.get("stop_loss") or res.get("stop")
        tp_val = res.get("tp") or res.get("take_profit") or res.get("takeprofit")
        entry_point = res.get("entry_point") or (res.get("precision_entry") or {}).get("entry_price") or (res.get("precision_entry") or {}).get("entry_point")
        entry_time = res.get("entry_time_sast") or res.get("entry_time") or (res.get("precision_entry") or {}).get("entry_time_sast")
        confidence = res.get("confidence") or (res.get("precision_entry") or {}).get("confidence")
        retest_info = (res.get("context") or {}).get("retest") or res.get("retest") or (res.get("precision_entry") or {}).get("reason")
        
        # **GUI GUARANTEE:** Prioritize the final, GOAT-adjusted lot size.
        lot_reco = res.get("lot_size_recommendation_goat") or res.get("lot_size_recommendation") or res.get("lot")

        ctx = res.get("context") or {}
        ep = ctx.get("entry_plan") or {}
        if sl_val is None:
            sl_val = ep.get("sl")
        if tp_val is None:
            tp_val = ep.get("tp2") or ep.get("tp1")
        if entry_point is None:
            entry_point = ep.get("entry_zone")
        if entry_time is None:
            entry_time = ctx.get("entry_time")
        if price is None:
            price = (ctx.get("details") or {}).get("last_close")
        if atr is None:
            atr = (ctx.get("details") or {}).get("atr")
        if lot_reco is None:
            lot_reco = ep.get("size_suggestion")

        news_status = res.get("news_status") or (res.get("news") or {}).get("status") or "unknown"
        headlines = (res.get("news") or {}).get("headlines") or []
        liquidity = res.get("liquidity") or {}
        liquidity_score = liquidity.get("liquidity_score")
        micro = res.get("micro_confirmation")
        structure = res.get("structure_change") or res.get("structure")
        advanced = res.get("advanced")
        context_block = res.get("context")

        # NEW: PREX / TEO / QEF meta (if provided by gui_bridge / orchestrator)
        prex = res.get("prex") or {}
        teo = res.get("teo") or {}
        qef = res.get("qef") or {}

        prex_regime = prex.get("regime")
        prex_reason = prex.get("reason")
        teo_bias = teo.get("direction_bias")
        teo_wait_candles = teo.get("wait_candles")
        teo_wait_minutes = teo.get("estimated_wait_minutes")
        teo_wait_condition = teo.get("wait_condition")
        teo_invalid_if = teo.get("invalid_if")
        qef_best = (qef.get("best_entry") or {}) if isinstance(qef, dict) else {}
        qef_edge = qef_best.get("edge")
        qef_price = qef_best.get("price")
        qef_sl = qef_best.get("sl")
        qef_tp = qef_best.get("tp")

        self.lbl_price.setText(f"Price: {fmt_price(price)}")
        self.lbl_atr.setText(f"ATR: {fmt_price(atr)}")
        self.lbl_news_status.setText(f"News: {fmt_any(news_status)}")
        self.lbl_liquidity.setText(f"Liquidity: {fmt_any(liquidity_score)}")
        self.lbl_lot.setText(f"Lot: {fmt_any(lot_reco)}")

        verdict_text = verdict if verdict is not None else "—"
        verdict_up = verdict_text.upper()
        if verdict_up.startswith("BUY"):
            verdict_color = "#00FF88"
        elif verdict_up.startswith("SELL"):
            verdict_color = "#FF5555"
        else:
            verdict_color = "#FFFFFF"
        self.lbl_signal.setText(verdict_text)
        self.lbl_signal.setStyleSheet(f"font-size:16px;font-weight:800;color:{verdict_color};")

        conf_color = "#00FF88" if verdict_up.startswith("BUY") else "#FF5555" if verdict_up.startswith("SELL") else "#FFFFFF"

        # PREX/TEO/QEF header snippet (shown at the very top of the report)
        quantum_header_parts = []
        if prex_regime:
            quantum_header_parts.append(f"PREX Regime: {esc(prex_regime)} ({esc(prex_reason)})")
        if teo_bias:
            quantum_header_parts.append(
                f"Bias: {esc(teo_bias)} | Wait: {teo_wait_candles} candles "
                f"(~{teo_wait_minutes} min) | Condition: {esc(teo_wait_condition or '')} | "
                f"Invalid if: {esc(teo_invalid_if or '')}"
            )
        if qef_edge is not None and qef_price is not None:
            quantum_header_parts.append(
                f"QEF Optimal Entry: {fmt_price(qef_price)} (edge {qef_edge:.3f}) "
                f"SL {fmt_price(qef_sl)} TP {fmt_price(qef_tp)}"
            )
        quantum_header_html = ""
        if quantum_header_parts:
            quantum_header_html = (
                "<div style='font-family:Consolas,monospace;font-size:12px;color:#b0c4ff;"
                "background:#050b18;border-radius:6px;padding:6px 8px;margin-bottom:6px;'>"
                "<b>Quantum Pre-Execution:</b><br>"
                + "<br>".join(esc(line) for line in quantum_header_parts)
                + "</div><br>"
            )

        header_html = (
            "<div style='font-family:Consolas,monospace;'>"
            "<b><span style='font-size:16px;color:#ffffff;'>"
            f"SL: <span style='color:#FF5555;'>{esc(fmt_price(sl_val))}</span> &nbsp;&nbsp;"
            f"TP: <span style='color:#00FF88;'>{esc(fmt_price(tp_val))}</span> &nbsp;&nbsp;"
            f"Entry Point: {esc(fmt_price(entry_point))} &nbsp;&nbsp;"
            f"Entry Time: {esc(fmt_any(entry_time))} &nbsp;&nbsp;"
            f"Lot: <span style='color:#9ad1ff;'>{esc(fmt_any(lot_reco))}</span> &nbsp;&nbsp;"
            f"Confidence: <span style='color:{conf_color};'>{esc(fmt_any(confidence))}</span> &nbsp;&nbsp;"
            f"Retest: {esc(fmt_any(retest_info))} &nbsp;&nbsp;"
            f"Verdict: <span style='color:{verdict_color};font-weight:900;'>{esc(verdict_text)}</span>"
            "</span></b></div><br>"
        )

        report_block = res.get("report", "")
        report_text = "\n".join(f"{k}: {v}" for k, v in report_block.items()) if isinstance(report_block, dict) else str(report_block or "")
        body_html = f"<pre style='font-family:Consolas,monospace;color:#e6eef8;white-space:pre-wrap;word-wrap:break-word;'>{report_text}</pre>"

        extras = ""
        if advanced is not None:
            extras += "<br><b>--- Advanced ---</b><br><pre style='color:#e6eef8;white-space:pre-wrap;word-wrap:break-word;'>" + str(advanced) + "</pre>"
        if context_block is not None:
            extras += "<br><b>--- Context ---</b><br><pre style='color:#e6eef8;white-space:pre-wrap;word-wrap:break-word;'>" + str(context_block) + "</pre>"
        if micro:
            extras += "<br><b>Micro confirmation:</b> " + str(micro)
        if structure:
            extras += "<br><b>Structure/Liquidity:</b> " + str(structure)

        # Append PREX/TEO/QEF details into extras (so they're also in plain-text copy)
        if quantum_header_parts:
            extras += "<br><b>--- Quantum Pre-Execution ---</b><br><pre style='color:#b0c4ff;white-space:pre-wrap;word-wrap:break-word;'>" + "\n".join(quantum_header_parts) + "</pre>"

        try:
            self.text_report.setHtml(quantum_header_html + header_html + body_html + extras)
        except Exception:
            plain_quantum = ""
            if quantum_header_parts:
                plain_quantum = "Quantum Pre-Execution:\n" + "\n".join(quantum_header_parts) + "\n\n"
            self.text_report.setPlainText(
                plain_quantum +
                f"SL: {sl_val}  TP: {tp_val}  Entry: {entry_point}  Time: {entry_time}  Lot: {lot_reco}  Verdict: {verdict_text}\n\n{report_text}"
            )

        self.headlines_list.clear()
        for h in headlines:
            item = QtWidgets.QListWidgetItem(f"{h.get('headline') or ''}  [{h.get('impact') or ''}]  ({h.get('source') or ''})  {h.get('time') or ''}")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            item.setForeground(QtGui.QColor("#e6eef8"))
            self.headlines_list.addItem(item)
        if not headlines:
            self.headlines_list.addItem(QtWidgets.QListWidgetItem("No recent headlines"))

        self.lbl_status.setText("Analysis complete - Full orchestration executed")

        self._last_verdict = verdict_up
        self._last_sl = sl_val
        self._last_tp = tp_val
        self._last_entry = entry_point
        self._last_lot = lot_reco

        if (verdict_up.startswith("BUY") or verdict_up.startswith("SELL")) and all(
            self._is_valid_number(x) for x in [sl_val, tp_val, entry_point, lot_reco]
        ):
            self.btn_confirm.setEnabled(True)
        else:
            self.btn_confirm.setEnabled(False)

    def on_worker_error(self, err):
        if self._overlay:
            self._overlay.hide_overlay()
        self.btn_analyze.setEnabled(True)
        self.show_error(f"Worker error: {err}")

    def _is_valid_number(self, v):
        try:
            return v is not None and v != "" and float(v) == float(v)
        except Exception:
            return False

    def _parse_entry_value(self, raw_entry, is_buy):
        if raw_entry is None:
            return None
        if isinstance(raw_entry, (int, float)):
            return float(raw_entry)
        s = str(raw_entry).strip()
        rng = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\s*(?:-|–|to|TO)\s*([-+]?\d+(?:\.\d+)?)\s*$", s)
        if rng:
            a = float(rng.group(1)); b = float(rng.group(2))
            low, high = (a, b) if a <= b else (b, a)
            return low if is_buy else high
        try:
            return float(s)
        except Exception:
            return None

    def _pre_normalize_lot(self, symbol, lot):
        if mt5 is None:
            return lot, False
        try:
            info = mt5.symbol_info(symbol)
        except Exception:
            info = None
        if not info:
            return lot, False
        vol_min = getattr(info, "volume_min", 0.01)
        vol_step = getattr(info, "volume_step", 0.01)
        vol_max = getattr(info, "volume_max", None)
        if lot < vol_min:
            lot_norm = vol_min
        else:
            steps = int((lot - vol_min + 1e-12) / max(vol_step, 1e-12))
            lot_norm = vol_min + steps * vol_step
            if vol_max and lot_norm > vol_max:
                lot_norm = vol_max
        if abs(lot_norm - lot) > 1e-9:
            return lot_norm, True
        return lot, False

    def on_confirm_trade_clicked(self):
        if place_pending_order is None:
            self.show_error("Trade executor not available.")
            return

        symbol = self._last_symbol or self.edit_pair.text().strip()
        if not symbol:
            self.show_error("No symbol available for trade.")
            return

        verdict_up = self._last_verdict or ""
        is_buy = verdict_up.startswith("BUY") or False
        is_sell = verdict_up.startswith("SELL") or False
        if not (is_buy or is_sell):
            self.show_error("Verdict not actionable.")
            return

        if not all(self._is_valid_number(x) for x in [self._last_sl, self._last_tp, self._last_entry, self._last_lot]):
            self.show_error("Missing SL/TP/Entry/Lot - cannot trade.")
            return

        entry_price = self._parse_entry_value(self._last_entry, is_buy)
        if entry_price is None:
            self.show_error("Cannot parse entry price.")
            return

        sl = float(self._last_sl)
        tp = float(self._last_tp)
        lot_original = float(self._last_lot)

        lot_pre, lot_pre_adjusted = self._pre_normalize_lot(symbol, lot_original)

        self.lbl_status.setText("Placing pending order...")
        first_result = None
        auto_volume_retry = False
        final_lot_sent = lot_pre

        try:
            first_result = place_pending_order(
                symbol=symbol,
                is_buy=is_buy,
                entry_price=entry_price,
                sl=sl,
                tp=tp,
                lot=lot_pre,
                expiry_minutes=180,
                magic=777,
            )
        except Exception as e:
            self.show_error(f"Trade placement exception: {e}")
            return

        if first_result and first_result.get("error") == "volume_invalid" and first_result.get("volume_normalized") is not None:
            suggested = first_result.get("volume_normalized")
            if isinstance(suggested, (int, float)) and suggested != lot_pre:
                auto_volume_retry = True
                final_lot_sent = float(suggested)
                try:
                    second_result = place_pending_order(
                        symbol=symbol,
                        is_buy=is_buy,
                        entry_price=entry_price,
                        sl=sl,
                        tp=tp,
                        lot=final_lot_sent,
                        expiry_minutes=180,
                        magic=777,
                    )
                    first_result["auto_retry_initial_error"] = first_result.get("error")
                    first_result = second_result
                except Exception as e:
                    first_result["auto_retry_exception"] = str(e)

        result = first_result or {}

        if result.get("ok"):
            self.lbl_status.setText("Trade placed (pending) ✅")
        else:
            self.lbl_status.setText(f"Trade failed: {result.get('error') or result.get('retcode')}")

        exec_lines = [
            "",
            "--- Execution Summary ---",
            f"symbol: {symbol}",
            f"verdict: {verdict_up}",
            f"is_buy: {is_buy}",
            f"entry_price: {entry_price}",
            f"sl: {sl}",
            f"tp: {tp}",
            f"lot_original: {lot_original}",
            f"lot_sent_initial: {lot_pre}",
            f"pre_normalized: {lot_pre_adjusted}",
            f"auto_volume_retry: {auto_volume_retry}",
            f"lot_final_sent: {final_lot_sent}",
            f"pending_type: {result.get('pending_type')}",
            f"resolved_symbol: {result.get('resolved_symbol')}",
            f"retcode: {result.get('retcode')}",
            f"error: {result.get('error')}",
            f"comment: {result.get('comment')}",
            f"root_cause: {', '.join(result.get('root_cause') or [])}",
            "--- Preflight ---",
            f"terminal_info: {result.get('preflight',{}).get('terminal_info')}",
            f"account_info: {result.get('preflight',{}).get('account_info')}",
            f"raw_symbol_info: {result.get('preflight',{}).get('raw_symbol_info')}",
        ]
        sc = result.get("symbol_constraints") or {}
        if sc:
            exec_lines.append(f"symbol_constraints: min={sc.get('volume_min')} step={sc.get('volume_step')} max={sc.get('volume_max')}")
        if result.get("volume_normalized") is not None:
            exec_lines.append(f"executor_volume_normalized: {result.get('volume_normalized')}")
        if "auto_retry_initial_error" in result:
            exec_lines.append(f"initial_error_before_retry: {result.get('auto_retry_initial_error')}")
        if "auto_retry_exception" in result:
            exec_lines.append(f"auto_retry_exception: {result.get('auto_retry_exception')}")

        alts = result.get("alt_suggestions") or []
        if alts:
            exec_lines.append("--- Alt Suggestions (name | trade_mode | min | step | visible) ---")
            for s in alts[:20]:
                exec_lines.append(f"{s.get('name')} | {s.get('trade_mode')} | {s.get('volume_min')} | {s.get('volume_step')} | {s.get('visible')}")

        debug_log = result.get("debug_log")
        if debug_log:
            exec_lines.append("--- Debug Log ---")
            exec_lines.extend(debug_log)

        existing = self.text_report.toPlainText()
        self.text_report.setPlainText(existing + "\n" + "\n".join(exec_lines))

    def on_list_symbols(self):
        if list_symbols is None:
            self.show_error("list_symbols not available in data.py")
            return
        try:
            syms = list_symbols()
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Available Symbols")
            v = QtWidgets.QVBoxLayout(dlg)
            te = QtWidgets.QTextEdit()
            te.setReadOnly(True)
            te.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
            te.setStyleSheet("color:white;background:#0f1724;font-size:14px;")
            te.setPlainText("\n".join(syms))
            v.addWidget(te)
            btn = QtWidgets.QPushButton("Close")
            btn.clicked.connect(dlg.close)
            v.addWidget(btn)
            dlg.resize(600, 400)
            dlg.exec_()
        except Exception as e:
            self.show_error(str(e))

    def copy_report(self):
        try:
            txt = self.text_report.toPlainText()
            headlines_texts = [self.headlines_list.item(i).text() for i in range(self.headlines_list.count())]
            if headlines_texts:
                txt += "\n\nHeadlines:\n" + "\n".join(headlines_texts)
            lot_label = self.lbl_lot.text()
            if lot_label and "Lot:" in lot_label:
                txt = lot_label + "\n\n" + txt
            QtWidgets.QApplication.clipboard().setText(txt)
            self.lbl_status.setText("Report copied to clipboard")
        except Exception:
            self.lbl_status.setText("Copy failed")

    def show_error(self, message: str):
        self.lbl_status.setText(f"Error: {message}")
        dlg = QtWidgets.QMessageBox(self)
        dlg.setIcon(QtWidgets.QMessageBox.Warning)
        dlg.setWindowTitle("Error")
        dlg.setText(message)
        dlg.exec_()


def main():
    app = QtWidgets.QApplication(sys.argv)
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    win = FXGui()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
