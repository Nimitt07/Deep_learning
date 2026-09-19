"""
Generates Group12_Assignment1_report.pdf directly with reportlab (no LaTeX compiler
available on this machine). Mirrors the structure of the provided thesis-style .tex
template (cover page, table of contents, chaptered body) but trimmed down for a short
programming-assignment report rather than a full thesis.
"""

import os
import sys

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image as RLImage, KeepTogether, NextPageTemplate,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfgen import canvas as canvas_mod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_reports import parse_classification_report, parse_regression_report

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CODE_DIR = r"C:\Deep_learning\Group12_Assignment1_code"
OUT_DIR = os.path.join(CODE_DIR, "outputs")
EQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "equations")
LOGO_PATH = r"C:\Users\neera\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\F0050E69D51C921F2CB685C5C88827E1E5DD6EC3\transfers\2026-33\intro_to_AI_Course_Project_report\images\IITDh_logo-tr.png"
OUT_PDF = r"C:\Deep_learning\Group12_Assignment1_report.pdf"

GROUP_NUM = "12"
TITLE = "Perceptron-Based Classification and Regression"
SUBTITLE = "CS601T: Deep Learning \u2013 Programming Assignment I"
STUDENTS = [
    ("Dattaraj Balkrishna Saudagar", "CS24BT037"),
    ("Onkar Avinash Mulje", "CS24BT059"),
    ("Nimitt Jain", "CS24BT048"),
]
INSTITUTE_LINES = [
    "Department of Computer Science and Engineering",
    "Indian Institute of Technology Dharwad",
]
MONTH_YEAR = "August 2026"

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", fontSize=20, leading=26, alignment=TA_CENTER, spaceAfter=6, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="ReportSubtitle", fontSize=14, leading=18, alignment=TA_CENTER, spaceAfter=6, fontName="Helvetica"))
styles.add(ParagraphStyle(name="CoverText", fontSize=12, leading=16, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="ChapterHeading", fontSize=16, leading=20, spaceBefore=18, spaceAfter=10, fontName="Helvetica-Bold", textColor=colors.HexColor("#1a1a1a")))
styles.add(ParagraphStyle(name="SectionHeading", fontSize=13, leading=17, spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="SubsectionHeading", fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4, fontName="Helvetica-Bold", textColor=colors.HexColor("#333333")))
styles.add(ParagraphStyle(name="Body", fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=6))
styles.add(ParagraphStyle(name="Caption", fontSize=9, leading=12, alignment=TA_CENTER, fontName="Helvetica-Oblique", textColor=colors.HexColor("#444444"), spaceBefore=2, spaceAfter=10))
styles.add(ParagraphStyle(name="TableCell", fontSize=9, leading=11, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="TableHeader", fontSize=9, leading=11, alignment=TA_CENTER, fontName="Helvetica-Bold", textColor=colors.white))
styles.add(ParagraphStyle(name="AbstractBody", fontSize=10.5, leading=15, alignment=TA_JUSTIFY))

TOC_STYLE_1 = ParagraphStyle(name="TOCHeading1", fontSize=11, leading=16, fontName="Helvetica-Bold")
TOC_STYLE_2 = ParagraphStyle(name="TOCHeading2", fontSize=10, leading=14, leftIndent=14, fontName="Helvetica")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def P(text, style="Body"):
    return Paragraph(text, styles[style])


def img_scaled(path, max_w, max_h=None):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    ratio = w / h
    draw_w = max_w
    draw_h = draw_w / ratio
    if max_h is not None and draw_h > max_h:
        draw_h = max_h
        draw_w = draw_h * ratio
    return RLImage(path, width=draw_w, height=draw_h)


def figure(path, caption, max_w=CONTENT_W * 0.75):
    im = img_scaled(path, max_w)
    im.hAlign = "CENTER"
    return KeepTogether([im, Spacer(1, 2), P(caption, "Caption")])


def figure_row(items, max_w_each=CONTENT_W / 2 - 0.3 * cm):
    """items: list of (path, caption); lays out 2 per row in a borderless table."""
    cells = []
    row = []
    for path, caption in items:
        im = img_scaled(path, max_w_each)
        cell = [im, Spacer(1, 2), P(caption, "Caption")]
        row.append(cell)
        if len(row) == 2:
            cells.append(row)
            row = []
    if row:
        row.append("")
        cells.append(row)
    t = Table(cells, colWidths=[max_w_each + 0.3 * cm] * 2)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def eq(name, scale=1.0):
    path = os.path.join(EQ_DIR, f"{name}.png")
    im = img_scaled(path, max_w=CONTENT_W * 0.85 * scale)
    im.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 4), im, Spacer(1, 6)])


def confusion_matrix_table(result):
    classes = result["classes"]
    cm = result["confusion_matrix"]
    header = [P("True \\ Pred", "TableHeader")] + [P(f"Class {c}", "TableHeader") for c in classes]
    data = [header]
    for i, c in enumerate(classes):
        row = [P(f"Class {c}", "TableHeader")] + [P(str(cm[i][j]), "TableCell") for j in range(len(classes))]
        data.append(row)
    n = len(classes)
    col_w = (CONTENT_W * 0.6) / (n + 1)
    t = Table(data, colWidths=[col_w] * (n + 1))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f5597")),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#2f5597")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def metrics_table(result):
    classes = result["classes"]
    header = [P(h, "TableHeader") for h in ["Class", "Precision", "Recall", "F-measure"]]
    data = [header]
    for c in classes:
        pc = result["per_class"][c]
        data.append([
            P(f"Class {c}", "TableCell"),
            P(f"{pc['precision']:.4f}", "TableCell"),
            P(f"{pc['recall']:.4f}", "TableCell"),
            P(f"{pc['f1']:.4f}", "TableCell"),
        ])
    mean = result["mean"]
    data.append([
        P("Mean", "TableHeader"),
        P(f"{mean['precision']:.4f}", "TableCell"),
        P(f"{mean['recall']:.4f}", "TableCell"),
        P(f"{mean['f1']:.4f}", "TableCell"),
    ])
    col_w = CONTENT_W * 0.6 / 4
    t = Table(data, colWidths=[col_w] * 4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f5597")),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#d9e2f3")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def rmse_table(result):
    header = [P(h, "TableHeader") for h in ["Split", "RMSE", "%RMSE"]]
    data = [header,
            [P("Training", "TableCell"), P(f"{result['rmse_train']:.4f}", "TableCell"), P(f"{result['pct_rmse_train']:.2f}%", "TableCell")],
            [P("Test", "TableCell"), P(f"{result['rmse_test']:.4f}", "TableCell"), P(f"{result['pct_rmse_test']:.2f}%", "TableCell")]]
    col_w = CONTENT_W * 0.5 / 3
    t = Table(data, colWidths=[col_w] * 3)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f5597")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    t.hAlign = "CENTER"
    return t


# ---------------------------------------------------------------------------
# Document template with TOC + page numbers + chapter bookmarks
# ---------------------------------------------------------------------------

class ReportDocTemplate(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            text = flowable.getPlainText()
            if style_name == "ChapterHeading":
                key = f"ch-{abs(hash(text))}"
                self.canv.bookmarkPage(key)
                self.notify("TOCEntry", (0, text, self.page, key))
            elif style_name == "SectionHeading":
                key = f"sec-{abs(hash(text))}"
                self.canv.bookmarkPage(key)
                self.notify("TOCEntry", (1, text, self.page, key))


def make_frame_body():
    return Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN, id="body")


def on_page_body(canvas_obj, doc_obj):
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 9)
    canvas_obj.setFillColor(colors.grey)
    canvas_obj.drawCentredString(PAGE_W / 2, MARGIN * 0.5, str(canvas_obj.getPageNumber()))
    canvas_obj.restoreState()


def on_page_blank(canvas_obj, doc_obj):
    pass


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------

def build_cover(story):
    story.append(Spacer(1, 1.2 * cm))
    story.append(P(TITLE, "ReportTitle"))
    story.append(P(SUBTITLE, "ReportSubtitle"))
    story.append(Spacer(1, 1.0 * cm))
    story.append(P(f"<b>Group {GROUP_NUM}</b>", "CoverText"))
    story.append(Spacer(1, 0.6 * cm))
    story.append(P("<i>Submitted by</i>", "CoverText"))
    story.append(Spacer(1, 0.2 * cm))
    for name, roll in STUDENTS:
        story.append(P(f"<b>{name}</b> (Roll No.: {roll})", "CoverText"))
    story.append(Spacer(1, 1.2 * cm))
    if os.path.exists(LOGO_PATH):
        logo = img_scaled(LOGO_PATH, max_w=3.4 * cm)
        logo.hAlign = "CENTER"
        story.append(logo)
        story.append(Spacer(1, 0.5 * cm))
    for line in INSTITUTE_LINES:
        story.append(P(f"<b>{line}</b>", "CoverText"))
    story.append(Spacer(1, 0.2 * cm))
    story.append(P(MONTH_YEAR, "CoverText"))
    story.append(PageBreak())


def build_toc(story):
    toc = TableOfContents()
    toc.levelStyles = [TOC_STYLE_1, TOC_STYLE_2]
    story.append(P("Contents", "ChapterHeading"))
    story.append(toc)
    story.append(PageBreak())


# ---------------------------------------------------------------------------
# Content chapters
# ---------------------------------------------------------------------------

def build_abstract(story):
    story.append(P("Abstract", "ChapterHeading"))
    story.append(P(
        "This report presents the design, implementation, and evaluation of a single-layer "
        "perceptron for both classification and regression tasks, implemented entirely from "
        "scratch (without using any perceptron, neural network, or gradient-descent library). "
        "For classification, perceptrons with logistic and tan-hyperbolic activation functions "
        "were trained using a one-against-one strategy on a linearly separable (LS) and a "
        "non-linearly separable (NLS) three-class, two-dimensional dataset. For regression, a "
        "perceptron with a linear activation function was trained on a univariate and a "
        "bivariate dataset. All models were trained using the online gradient-descent learning "
        "rule. Performance is reported using confusion matrices, accuracy, class-wise and mean "
        "precision/recall/F-measure for classification, and RMSE / %RMSE for regression, all "
        "computed using custom implementations. Results show perfect separation on the linearly "
        "separable data and the expected performance degradation on the non-linearly separable "
        "and non-linear regression data, illustrating the fundamental limitation of a single "
        "linear decision boundary.",
        "AbstractBody"))
    story.append(PageBreak())


def build_chapter1_intro(story):
    story.append(P("1.&nbsp;&nbsp;Introduction", "ChapterHeading"))
    story.append(P(
        "The perceptron is the simplest artificial neuron: it computes a weighted sum of its "
        "inputs (the net input), passes this sum through a non-linear activation function, and "
        "produces a single scalar output. Despite its simplicity, the perceptron is the "
        "fundamental building block of larger neural networks, and understanding its learning "
        "rule and its limitations motivates the need for multi-layer architectures.", "Body"))
    story.append(P(
        "This assignment implements the perceptron learning algorithm from first principles "
        "and applies it to two kinds of tasks:", "Body"))
    story.append(P(
        "<b>Classification:</b> a 3-class, 2-dimensional problem is solved using a one-against-one "
        "ensemble of perceptrons, once with a logistic activation function and once with a "
        "tan-hyperbolic (tanh) activation function, on two datasets \u2014 one linearly separable "
        "(LS) and one non-linearly separable (NLS).", "Body"))
    story.append(P(
        "<b>Regression:</b> a perceptron with a linear activation function is trained to predict "
        "a continuous target from a univariate input and, separately, from a bivariate input.", "Body"))
    story.append(P(
        "In both cases the weights are updated using the online (per-sample) gradient-descent "
        "rule that minimizes the mean-squared error between the perceptron's output and the "
        "target, and every evaluation metric (confusion matrix, precision, recall, F-measure, "
        "RMSE, %RMSE) is implemented from scratch as required by the assignment.", "Body"))
    story.append(PageBreak())


def build_chapter2_methodology(story):
    story.append(P("2.&nbsp;&nbsp;Methodology", "ChapterHeading"))

    story.append(P("2.1&nbsp;&nbsp;Perceptron Model", "SectionHeading"))
    story.append(P(
        "Each perceptron computes a net input as a weighted sum of the input features plus a "
        "bias, and applies an activation function f to obtain its output o = f(net):", "Body"))
    story.append(eq("net_input", scale=0.5))
    story.append(P("Three activation functions are used in this assignment:", "Body"))
    story.append(P("<b>Logistic (sigmoid)</b> \u2014 used for classification, output in (0, 1):", "Body"))
    story.append(eq("logistic", scale=0.85))
    story.append(P("<b>Tan-hyperbolic (tanh)</b> \u2014 used for classification, output in (\u22121, 1):", "Body"))
    story.append(eq("tanh", scale=0.7))
    story.append(P("<b>Linear</b> \u2014 used for regression, output unbounded:", "Body"))
    story.append(eq("linear", scale=0.55))

    story.append(P("2.2&nbsp;&nbsp;Gradient-Descent Learning Rule", "SectionHeading"))
    story.append(P(
        "Training minimizes the mean-squared error between target t and output o over the "
        "N training samples:", "Body"))
    story.append(eq("error", scale=0.45))
    story.append(P(
        "Differentiating this error with respect to each weight gives the online (per-sample) "
        "gradient-descent update rule, applied after every training example with learning rate "
        "\u03b7:", "Body"))
    story.append(eq("update_rule", scale=1.0))
    story.append(P(
        "For the logistic and tanh activations the targets are encoded as {0,1} and {\u22121,+1} "
        "respectively; for the linear activation used in regression, f'(net) = 1 and the target "
        "is the real-valued output directly. Weights are initialized to small random values and "
        "the dataset order is reshuffled every epoch.", "Body"))

    story.append(P("2.3&nbsp;&nbsp;One-Against-One Multi-Class Strategy", "SectionHeading"))
    story.append(P(
        "Since a single perceptron only produces a binary decision, the 3-class classification "
        "problem is solved with three pairwise perceptrons: Class1-vs-Class2, Class1-vs-Class3, "
        "and Class2-vs-Class3, each trained only on the samples belonging to that pair of "
        "classes. At prediction time, all three pairwise classifiers vote for one of their two "
        "classes, and the class with the majority of votes is chosen as the final label; ties "
        "(possible when each classifier votes for a different class) are broken using the sum "
        "of the winning classifiers' confidence scores (|output \u2212 decision threshold|).", "Body"))

    story.append(P("2.4&nbsp;&nbsp;Evaluation Metrics (implemented from scratch)", "SectionHeading"))
    story.append(P(
        "For classification, a confusion matrix CM is built by counting true-label vs. "
        "predicted-label pairs on the test set. From it, per-class precision and recall are "
        "computed as:", "Body"))
    story.append(eq("precision", scale=0.35))
    story.append(eq("recall", scale=0.35))
    story.append(P("and the per-class F-measure is their harmonic mean:", "Body"))
    story.append(eq("fmeasure", scale=0.55))
    story.append(P(
        "Accuracy is the fraction of correctly classified test samples, and the mean precision, "
        "mean recall and mean F-measure are simple averages of the per-class values. For "
        "regression, the root-mean-squared error and percentage RMSE (RMSE normalized by the "
        "mean of the target values) are used:", "Body"))
    story.append(eq("rmse", scale=0.4))
    story.append(eq("pct_rmse", scale=0.35))
    story.append(PageBreak())


def build_chapter3_setup(story):
    story.append(P("3.&nbsp;&nbsp;Experimental Setup", "ChapterHeading"))

    story.append(P("3.1&nbsp;&nbsp;Datasets", "SectionHeading"))
    data = [
        [P(h, "TableHeader") for h in ["Task", "Dataset", "Size", "Description"]],
        [P("Classification", "TableCell"), P("LS (Dataset 1)", "TableCell"), P("3 \u00d7 500 = 1500", "TableCell"), P("2-D, linearly separable", "TableCell")],
        [P("Classification", "TableCell"), P("NLS (Dataset 2)", "TableCell"), P("3 \u00d7 500 = 1500", "TableCell"), P("2-D, non-linearly separable", "TableCell")],
        [P("Regression", "TableCell"), P("Univariate (Dataset 1)", "TableCell"), P("1000", "TableCell"), P("1-D input, 1-D output", "TableCell")],
        [P("Regression", "TableCell"), P("Bivariate (Dataset 2)", "TableCell"), P("10200", "TableCell"), P("2-D input, 1-D output", "TableCell")],
    ]
    col_w = [CONTENT_W * f for f in (0.2, 0.25, 0.2, 0.35)]
    t = Table(data, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f5597")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(P("3.2&nbsp;&nbsp;Train / Test Split", "SectionHeading"))
    story.append(P(
        "For every dataset, each class (classification) or the whole dataset (regression) is "
        "split into 70% training data and 30% test data using a fixed random seed. The split is "
        "performed exactly once by a dedicated data-preparation step and the resulting indices "
        "are cached to disk; every subsequent training run (for any activation function or "
        "model configuration) loads this same cached split, so the train/test partition never "
        "changes when switching between models.", "Body"))

    story.append(P("3.3&nbsp;&nbsp;Hyperparameters", "SectionHeading"))
    hdata = [
        [P(h, "TableHeader") for h in ["Experiment", "Learning Rate", "Max Epochs"]],
        [P("Classification (LS, NLS)", "TableCell"), P("0.05", "TableCell"), P("300", "TableCell")],
        [P("Regression (Univariate)", "TableCell"), P("0.01", "TableCell"), P("200", "TableCell")],
        [P("Regression (Bivariate)", "TableCell"), P("0.01", "TableCell"), P("200", "TableCell")],
    ]
    col_w2 = [CONTENT_W * f for f in (0.45, 0.275, 0.275)]
    t2 = Table(hdata, colWidths=col_w2)
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f5597")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)
    story.append(PageBreak())


def build_classification_results(story, name_slug, display_name, chapter_num, extra_note=""):
    story.append(P(f"{chapter_num}&nbsp;&nbsp;Classification Results \u2014 {display_name}", "SectionHeading"))
    if extra_note:
        story.append(P(extra_note, "Body"))

    for activation, act_label in (("logistic", "Logistic Activation"), ("tanh", "Tan-Hyperbolic Activation")):
        act_dir = os.path.join(OUT_DIR, f"classification_{name_slug}", activation)
        story.append(P(f"{chapter_num}.{'1' if activation == 'logistic' else '2'}&nbsp;&nbsp;{act_label}", "SubsectionHeading"))

        story.append(figure(os.path.join(act_dir, "error_vs_epochs.png"),
                             f"Average error vs. epochs for each one-against-one pair ({display_name}, {act_label}).",
                             max_w=CONTENT_W * 0.7))

        pair_items = [
            (os.path.join(act_dir, "decision_region_1v2.png"), "Class 1 vs Class 2"),
            (os.path.join(act_dir, "decision_region_1v3.png"), "Class 1 vs Class 3"),
            (os.path.join(act_dir, "decision_region_2v3.png"), "Class 2 vs Class 3"),
        ]
        story.append(figure_row(pair_items))
        story.append(figure(os.path.join(act_dir, "decision_region_combined.png"),
                             f"Combined decision regions after one-against-one voting ({display_name}, {act_label}), training data only.",
                             max_w=CONTENT_W * 0.55))

        result = parse_classification_report(os.path.join(act_dir, "evaluation_report.txt"))
        story.append(P(f"<b>Confusion matrix (test data, {act_label}):</b>", "Body"))
        story.append(confusion_matrix_table(result))
        story.append(Spacer(1, 8))
        story.append(P(f"<b>Test accuracy: {result['accuracy']:.2f}%</b>", "Body"))
        story.append(metrics_table(result))
        story.append(Spacer(1, 12))


def build_regression_results(story, name_slug, display_name, chapter_num, axis_note):
    story.append(P(f"{chapter_num}&nbsp;&nbsp;Regression Results \u2014 {display_name}", "SectionHeading"))
    r_dir = os.path.join(OUT_DIR, f"regression_{name_slug}")

    story.append(figure(os.path.join(r_dir, "error_vs_epochs.png"),
                         f"Average error vs. epochs ({display_name}).", max_w=CONTENT_W * 0.7))

    result = parse_regression_report(os.path.join(r_dir, "evaluation_report.txt"))
    story.append(P("<b>RMSE and %RMSE:</b>", "Body"))
    story.append(rmse_table(result))
    story.append(Spacer(1, 10))

    story.append(P(f"<b>Model output superimposed on target output</b> ({axis_note}):", "Body"))
    story.append(figure_row([
        (os.path.join(r_dir, "model_vs_target_train.png"), "Training data"),
        (os.path.join(r_dir, "model_vs_target_test.png"), "Test data"),
    ]))

    story.append(P("<b>Scatter plot: target output (x-axis) vs. model output (y-axis)</b>:", "Body"))
    story.append(figure_row([
        (os.path.join(r_dir, "scatter_target_vs_model_train.png"), "Training data"),
        (os.path.join(r_dir, "scatter_target_vs_model_test.png"), "Test data"),
    ]))
    story.append(Spacer(1, 10))


def build_chapter4_results(story):
    story.append(P("4.&nbsp;&nbsp;Results", "ChapterHeading"))
    build_classification_results(story, "ls", "Dataset 1 (Linearly Separable)", "4.1")
    story.append(PageBreak())
    build_classification_results(story, "nls", "Dataset 2 (Non-Linearly Separable)", "4.2")
    story.append(PageBreak())
    build_regression_results(story, "univariate", "Dataset 1 (Univariate)", "4.3",
                              "x-axis: input x, y-axis: target / model output")
    story.append(PageBreak())
    build_regression_results(story, "bivariate", "Dataset 2 (Bivariate)", "4.4",
                              "x1, x2 axes: input values, y-axis: target / model output")
    story.append(PageBreak())


def build_chapter5_inferences(story):
    story.append(P("5.&nbsp;&nbsp;Inferences and Observations", "ChapterHeading"))

    story.append(P("5.1&nbsp;&nbsp;Classification", "SectionHeading"))
    story.append(P(
        "On the <b>linearly separable (LS)</b> dataset, all three one-against-one perceptrons "
        "converge to an average error close to zero within a small number of epochs, and the "
        "combined decision regions cleanly separate the three classes with straight-line "
        "boundaries. This yields 100% test accuracy, 100% precision, recall and F-measure for "
        "every class, for both the logistic and tanh activation functions \u2014 as expected, since "
        "the three classes are, by construction, separable by straight lines and a single "
        "perceptron per pair is sufficient to find such a boundary.", "Body"))
    story.append(P(
        "On the <b>non-linearly separable (NLS)</b> dataset, the three classes form curved, "
        "interleaved \u201cmoon-shaped\u201d clusters that cannot be separated by straight lines. "
        "Because each pairwise perceptron can only learn a linear decision boundary, the "
        "combined classifier achieves a lower test accuracy of about 92.9% for both activation "
        "functions, with Class 2 (the class sandwiched between the other two) showing the "
        "lowest recall and F-measure, since it is most often confused with its neighbours in "
        "the pairwise classifiers. The logistic and tanh activations produce almost identical "
        "accuracy and confusion patterns; the two activations differ mainly in their output "
        "range and in the shape of the error surface during training, but since both ultimately "
        "learn the same class of linear decision boundary, their final decision regions and "
        "test performance are very similar. This confirms that the performance gap between the "
        "LS and NLS datasets is caused by the linear nature of the perceptron's decision "
        "boundary rather than by the choice of activation function.", "Body"))

    story.append(P("5.2&nbsp;&nbsp;Regression", "SectionHeading"))
    story.append(P(
        "In both regression datasets the target output is a visibly non-linear function of the "
        "input(s): the univariate target follows a U-shaped (roughly quadratic, symmetric about "
        "the middle of the input range) curve, and the bivariate target grows in a curved, "
        "super-linear fashion with x1 and x2. Because the perceptron uses a linear activation "
        "function, it can only fit a straight line (univariate) or a flat plane (bivariate) to "
        "the data \u2014 it has no way to represent curvature.", "Body"))
    story.append(P(
        "For the univariate case, the best linear fit is nearly flat, since the target is "
        "symmetric and rises on both sides of the input range; this gives a relatively high "
        "%RMSE of around 42\u201344% on training and test data. The bivariate case shows a similar "
        "pattern with an even higher %RMSE of around 56\u201357%, reflecting the stronger curvature "
        "of that target surface. In both cases the average-error-vs-epochs plot drops sharply "
        "within the first few epochs and then plateaus, showing that gradient descent quickly "
        "converges to the best possible linear fit \u2014 the remaining error is not due to "
        "under-training but is an irreducible bias caused by using a linear model for a "
        "non-linear relationship. The scatter plots of target vs. model output further confirm "
        "this: the points form a roughly horizontal band far from the ideal y = x line, "
        "consistent with the model consistently under/over-predicting for large deviations of "
        "the input from the middle of its range. Test-set RMSE and %RMSE are close to the "
        "corresponding training-set values in both datasets, which shows the perceptron is not "
        "overfitting \u2014 its limited capacity is simply insufficient to capture the non-linear "
        "trend in the data.", "Body"))
    story.append(PageBreak())


def build_chapter6_conclusion(story):
    story.append(P("6.&nbsp;&nbsp;Conclusion", "ChapterHeading"))
    story.append(P(
        "This assignment implemented the perceptron learning algorithm and its online "
        "gradient-descent training rule entirely from scratch, together with custom "
        "implementations of every evaluation metric used. Across four experiments \u2014 "
        "one-against-one classification with logistic and tanh activations on linearly and "
        "non-linearly separable data, and linear-activation regression on univariate and "
        "bivariate data \u2014 a consistent pattern emerges: whenever the true relationship between "
        "input and output is linear (or the classes are linearly separable), a single "
        "perceptron is entirely sufficient and achieves near-perfect performance. Whenever the "
        "true relationship is non-linear, a single perceptron is fundamentally limited by its "
        "linear decision boundary / linear output surface, and no amount of additional training "
        "epochs or change of activation function can close this gap. This directly motivates "
        "the use of multi-layer neural networks with non-linear hidden layers, which are able "
        "to combine multiple linear boundaries to approximate arbitrarily non-linear decision "
        "regions and regression surfaces.", "Body"))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def main():
    doc = ReportDocTemplate(
        OUT_PDF, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
        title="Group12 Assignment 1 Report", author=", ".join(n for n, _ in STUDENTS),
    )
    frame_body = make_frame_body()
    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[frame_body], onPage=on_page_blank),
        PageTemplate(id="Body", frames=[frame_body], onPage=on_page_body),
    ])

    story = []
    build_cover(story)
    story.append(NextPageTemplate("Body"))
    build_toc(story)
    build_abstract(story)
    build_chapter1_intro(story)
    build_chapter2_methodology(story)
    build_chapter3_setup(story)
    build_chapter4_results(story)
    build_chapter5_inferences(story)
    build_chapter6_conclusion(story)

    doc.multiBuild(story)
    print(f"Report written to {OUT_PDF}")


if __name__ == "__main__":
    main()
