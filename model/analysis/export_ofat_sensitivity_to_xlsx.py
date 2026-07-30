from __future__ import annotations

import csv
import json
import math
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


DEFAULT_RUN_FOLDER = Path("model/ofat_runs/ofat_sensitivity_20260422-121429")


def is_number(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def excel_column_name(column_number: int) -> str:
    name = ""
    while column_number > 0:
        column_number, remainder = divmod(column_number - 1, 26)
        name = chr(65 + remainder) + name
    return name


def cell_xml(row_index: int, column_index: int, value: object) -> str:
    cell_ref = f"{excel_column_name(column_index)}{row_index}"
    if value is None or value == "":
        return f'<c r="{cell_ref}"/>'

    text = str(value)
    if is_number(text):
        return f'<c r="{cell_ref}"><v>{text}</v></c>'

    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'


def worksheet_xml(rows: list[list[object]]) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "<sheetData>",
    ]
    for r_idx, row in enumerate(rows, start=1):
        parts.append(f'<row r="{r_idx}">')
        for c_idx, value in enumerate(row, start=1):
            parts.append(cell_xml(r_idx, c_idx, value))
        parts.append("</row>")
    parts.extend(["</sheetData>", "</worksheet>"])
    return "".join(parts)


def metric_name(column_name: str) -> str:
    return column_name.removeprefix("final_mean__")


def numeric_change(value: str, baseline: str) -> tuple[str, str]:
    if not (is_number(value) and is_number(baseline)):
        return "", ""

    value_num = float(value)
    baseline_num = float(baseline)
    absolute_change = repr(value_num - baseline_num)
    pct_change = ""
    if baseline_num != 0:
        pct_change = repr(((value_num - baseline_num) / baseline_num) * 100.0)
    return absolute_change, pct_change


def load_inputs(run_folder: Path) -> tuple[list[dict[str, str]], dict]:
    aggregated_path = run_folder / "aggregated.csv"
    metadata_path = run_folder / "metadata.json"

    with aggregated_path.open("r", encoding="utf-8", newline="") as handle:
        aggregated = list(csv.DictReader(handle))

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    return aggregated, metadata


def build_workbook_data(
    aggregated: list[dict[str, str]],
    metadata: dict,
) -> list[tuple[str, list[list[object]]]]:
    columns = list(aggregated[0].keys())
    metric_cols = [col for col in columns if col.startswith("final_mean__")]
    base_cols = [
        "parameter",
        "mode",
        "grid_index",
        "factor_pct",
        "factor_label",
        "param_value",
        "default_value",
        "replicates",
        "runtime_mean_seconds",
    ]

    baselines: dict[tuple[str, str], dict[str, str]] = {}
    for row in aggregated:
        if row["param_value"] == row["default_value"]:
            baselines.setdefault((row["parameter"], row["mode"]), row)

    readme_rows: list[list[object]] = [
        ["Bestand", "Waarde"],
        ["Batch", metadata["batch_name"]],
        ["Aangemaakt", metadata["created_at"]],
        ["Replicates per gridpunt", metadata["replicates_per_point"]],
        ["Modes", ", ".join(metadata["modes"])],
        ["Parameters", metadata["parameter_count"]],
        ["Beschrijving", "Geaggregeerde OFAT-tabellen op basis van gemiddelden over replicates."],
        ["Sheet aggregated_means", "Ruwe gemiddelden per parameter x mode x gridpunt."],
        ["Sheet effects_all", "Lange tabel met mean, baseline, absolute change en pct change voor alle metrics."],
        ["Sheet gini_by_mode", "Samenvatting van de Gini-index per parameterverandering en per mode."],
        ["Mode-sheets", "Per mode dezelfde effectentabel, filterbaar in Excel."],
    ]

    aggregated_rows: list[list[object]] = [base_cols + metric_cols]
    for row in aggregated:
        aggregated_rows.append([row[col] for col in base_cols + metric_cols])

    effect_header = [
        "parameter",
        "mode",
        "factor_label",
        "factor_pct",
        "param_value",
        "default_value",
        "metric",
        "mean_value",
        "baseline_value",
        "absolute_change",
        "pct_change",
    ]
    effects_all_rows: list[list[object]] = [effect_header]
    mode_rows: dict[str, list[list[object]]] = {mode: [effect_header] for mode in metadata["modes"]}

    for row in aggregated:
        baseline = baselines.get((row["parameter"], row["mode"]))
        for metric_col in metric_cols:
            mean_value = row[metric_col]
            baseline_value = baseline[metric_col] if baseline else ""
            absolute_change, pct_change = numeric_change(mean_value, baseline_value)
            effect_row = [
                row["parameter"],
                row["mode"],
                row["factor_label"],
                row["factor_pct"],
                row["param_value"],
                row["default_value"],
                metric_name(metric_col),
                mean_value,
                baseline_value,
                absolute_change,
                pct_change,
            ]
            effects_all_rows.append(effect_row)
            mode_rows[row["mode"]].append(effect_row)

    gini_header = ["parameter", "factor_label", "factor_pct", "param_value", "default_value"]
    for mode in metadata["modes"]:
        gini_header.extend(
            [
                f"gini_mean_{mode}",
                f"gini_baseline_{mode}",
                f"gini_abs_change_{mode}",
                f"gini_pct_change_{mode}",
            ]
        )

    grouped: dict[tuple[str, str, str, str, str], dict[str, dict[str, str]]] = {}
    for row in aggregated:
        key = (
            row["parameter"],
            row["factor_label"],
            row["factor_pct"],
            row["param_value"],
            row["default_value"],
        )
        grouped.setdefault(key, {})[row["mode"]] = row

    gini_rows: list[list[object]] = [gini_header]
    for key in sorted(grouped):
        parameter, factor_label, factor_pct, param_value, default_value = key
        row_out: list[object] = [parameter, factor_label, factor_pct, param_value, default_value]
        for mode in metadata["modes"]:
            mode_row = grouped[key].get(mode)
            if mode_row is None:
                row_out.extend(["", "", "", ""])
                continue
            baseline = baselines.get((parameter, mode))
            mean_value = mode_row["final_mean__gini_income"]
            baseline_value = baseline["final_mean__gini_income"] if baseline else ""
            absolute_change, pct_change = numeric_change(mean_value, baseline_value)
            row_out.extend([mean_value, baseline_value, absolute_change, pct_change])
        gini_rows.append(row_out)

    sheets: list[tuple[str, list[list[object]]]] = [
        ("README", readme_rows),
        ("aggregated_means", aggregated_rows),
        ("effects_all", effects_all_rows),
        ("gini_by_mode", gini_rows),
    ]
    for mode in metadata["modes"]:
        sheets.append((mode[:31], mode_rows[mode]))
    return sheets


def write_xlsx(sheets: list[tuple[str, list[list[object]]]], output_path: Path) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    workbook_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        "<sheets>",
    ]
    for idx, (name, _) in enumerate(sheets, start=1):
        workbook_xml.append(f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>')
    workbook_xml.extend(["</sheets>", "</workbook>"])

    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for idx in range(1, len(sheets) + 1):
        workbook_rels.append(
            '<Relationship '
            f'Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    workbook_rels.append("</Relationships>")

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for idx in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

    core_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""

    app_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("docProps/app.xml", app_xml)
        zf.writestr("xl/workbook.xml", "".join(workbook_xml))
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        for idx, (_, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", worksheet_xml(rows))


def export_ofat_sensitivity_to_xlsx(run_folder: Path = DEFAULT_RUN_FOLDER) -> Path:
    aggregated, metadata = load_inputs(run_folder)
    sheets = build_workbook_data(aggregated, metadata)
    output_path = run_folder / "ofat_sensitivity_tables.xlsx"
    write_xlsx(sheets, output_path)
    return output_path


if __name__ == "__main__":
    output = export_ofat_sensitivity_to_xlsx()
    print(f"Gemaakt: {output}")
