CS601T Deep Learning - Programming Assignment II - Group 12
LaTeX (Springer LNCS) source of the report
===========================================================

Files
  Group12_Assignment2_report.tex   the report (generated; do not edit by hand)
  llncs.cls, splncs04.bst          the AIFB2026 / Springer LNCS template
  IITDh_logo.png                   the institute logo used on the title
  figures/                         the 175 PNGs the report prints

Building
  Overleaf:  New Project -> Upload Project -> pick the zip of this folder,
             set the main document to Group12_Assignment2_report.tex, Recompile.
  Locally:   pdflatex Group12_Assignment2_report.tex     (run it TWICE, so the
             equation cross-references resolve)

Regenerating
  The .tex file is written by ../Group12_Assignment2_code/make_report_tex.py.
  Re-run the experiments, then:

      cd ../Group12_Assignment2_code
      python make_report_tex.py

  It reads outputs/<experiment>/results.json and copies in only the figures the
  report actually prints, so every number here follows the last run.

  The prose and the auto-generated inference text are shared with make_report.py
  (the reportlab build of the same report), so the two cannot disagree.

Before submitting
  Set MEMBERS in make_report_tex.py to the team's names, e.g.

      MEMBERS = r"Name One \and Name Two \and Name Three"

  then re-run make_report_tex.py. With MEMBERS empty the author line just says
  "Group 12".
