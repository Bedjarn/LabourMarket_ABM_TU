Add-Type -AssemblyName System.Drawing

$outDir = Join-Path (Get-Location) 'charts'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outPath = Join-Path $outDir 'h1_digital_intensity_eu_2025.png'

$euml = [char]0x00EB
$data = @(
    [pscustomobject]@{ Land = 'Finland'; Laag = 27; Hoog = 43; HeelHoog = 24 },
    [pscustomobject]@{ Land = 'Denemarken'; Laag = 27; Hoog = 39; HeelHoog = 26 },
    [pscustomobject]@{ Land = 'Nederland'; Laag = 27; Hoog = 42; HeelHoog = 20 },
    [pscustomobject]@{ Land = 'Zweden'; Laag = 36; Hoog = 35; HeelHoog = 15 },
    [pscustomobject]@{ Land = "Belgi$euml"; Laag = 29; Hoog = 37; HeelHoog = 18 },
    [pscustomobject]@{ Land = "Itali$euml"; Laag = 42; Hoog = 30; HeelHoog = 7 },
    [pscustomobject]@{ Land = 'Ierland'; Laag = 39; Hoog = 29; HeelHoog = 11 },
    [pscustomobject]@{ Land = 'Malta'; Laag = 37; Hoog = 34; HeelHoog = 8 },
    [pscustomobject]@{ Land = 'Luxemburg'; Laag = 38; Hoog = 29; HeelHoog = 9 },
    [pscustomobject]@{ Land = 'Spanje'; Laag = 34; Hoog = 30; HeelHoog = 11 },
    [pscustomobject]@{ Land = 'Cyprus'; Laag = 38; Hoog = 30; HeelHoog = 7 },
    [pscustomobject]@{ Land = 'Litouwen'; Laag = 32; Hoog = 29; HeelHoog = 13 },
    [pscustomobject]@{ Land = 'Duitsland'; Laag = 37; Hoog = 27; HeelHoog = 9 },
    [pscustomobject]@{ Land = 'Oostenrijk'; Laag = 38; Hoog = 26; HeelHoog = 9 },
    [pscustomobject]@{ Land = 'Estland'; Laag = 36; Hoog = 28; HeelHoog = 8 },
    [pscustomobject]@{ Land = 'EU-gemiddelde'; Laag = 35; Hoog = 27; HeelHoog = 9 },
    [pscustomobject]@{ Land = "Tsjechi$euml"; Laag = 31; Hoog = 28; HeelHoog = 11 },
    [pscustomobject]@{ Land = 'Frankrijk'; Laag = 36; Hoog = 26; HeelHoog = 7 },
    [pscustomobject]@{ Land = "Sloveni$euml"; Laag = 36; Hoog = 23; HeelHoog = 7 },
    [pscustomobject]@{ Land = 'Portugal'; Laag = 36; Hoog = 21; HeelHoog = 7 },
    [pscustomobject]@{ Land = 'Hongarije'; Laag = 33; Hoog = 21; HeelHoog = 6 },
    [pscustomobject]@{ Land = 'Polen'; Laag = 32; Hoog = 21; HeelHoog = 5 },
    [pscustomobject]@{ Land = 'Letland'; Laag = 31; Hoog = 20; HeelHoog = 7 },
    [pscustomobject]@{ Land = "Kroati$euml"; Laag = 29; Hoog = 21; HeelHoog = 7 },
    [pscustomobject]@{ Land = 'Slowakije'; Laag = 32; Hoog = 20; HeelHoog = 5 },
    [pscustomobject]@{ Land = 'Griekenland'; Laag = 31; Hoog = 19; HeelHoog = 6 },
    [pscustomobject]@{ Land = "Roemeni$euml"; Laag = 28; Hoog = 14; HeelHoog = 2 },
    [pscustomobject]@{ Land = 'Bulgarije'; Laag = 26; Hoog = 11; HeelHoog = 2 }
) | ForEach-Object {
    $_ | Add-Member -NotePropertyName Totaal -NotePropertyValue ($_.Laag + $_.Hoog + $_.HeelHoog) -PassThru
}

$width = 1600
$height = 1300
$left = 245
$right = 145
$top = 120
$bottom = 135
$plotWidth = $width - $left - $right
$plotHeight = $height - $top - $bottom
$barGap = 8
$barHeight = [Math]::Floor(($plotHeight - (($data.Count - 1) * $barGap)) / $data.Count)

$colors = @{
    Text = [System.Drawing.ColorTranslator]::FromHtml('#333333')
    Muted = [System.Drawing.ColorTranslator]::FromHtml('#666666')
    Grid = [System.Drawing.ColorTranslator]::FromHtml('#D9DEE3')
    Laag = [System.Drawing.ColorTranslator]::FromHtml('#BFC9D1')
    Hoog = [System.Drawing.ColorTranslator]::FromHtml('#005A9C')
    HeelHoog = [System.Drawing.ColorTranslator]::FromHtml('#E08600')
    White = [System.Drawing.Color]::White
}

$bmp = New-Object System.Drawing.Bitmap $width, $height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$g.Clear($colors.White)

$fontTitle = New-Object System.Drawing.Font 'DejaVu Sans', 18, ([System.Drawing.FontStyle]::Bold)
$font = New-Object System.Drawing.Font 'DejaVu Sans', 12
$fontBold = New-Object System.Drawing.Font 'DejaVu Sans', 12, ([System.Drawing.FontStyle]::Bold)
$fontSmall = New-Object System.Drawing.Font 'DejaVu Sans', 10
$fontTiny = New-Object System.Drawing.Font 'DejaVu Sans', 9

$brushText = New-Object System.Drawing.SolidBrush $colors.Text
$brushMuted = New-Object System.Drawing.SolidBrush $colors.Muted
$penGrid = New-Object System.Drawing.Pen $colors.Grid, 1

$g.DrawString('Digitale intensiteitsindex bedrijven, EU 2025', $fontTitle, $brushText, 35, 30)
$g.DrawString('% van bedrijven met 10 tot 250 werknemers; totaal = laag + hoog + heel hoog', $fontSmall, $brushMuted, 36, 67)

foreach ($tick in 0, 20, 40, 60, 80, 100) {
    $x = $left + ($tick / 100) * $plotWidth
    $g.DrawLine($penGrid, [float]$x, [float]$top, [float]$x, [float]($top + $plotHeight))
    $label = "$tick%"
    $size = $g.MeasureString($label, $fontSmall)
    $g.DrawString($label, $fontSmall, $brushMuted, [float]($x - $size.Width / 2), [float]($top + $plotHeight + 13))
}

for ($i = 0; $i -lt $data.Count; $i++) {
    $row = $data[$i]
    $y = $top + $i * ($barHeight + $barGap)
    $labelFont = if ($row.Land -eq 'Nederland') { $fontBold } else { $font }
    $labelSize = $g.MeasureString($row.Land, $labelFont)
    $g.DrawString($row.Land, $labelFont, $brushText, [float]($left - $labelSize.Width - 14), [float]($y + ($barHeight - $labelSize.Height) / 2))

    $x = $left
    foreach ($segment in @(
        @{ Name = 'Laag'; Value = $row.Laag; Color = $colors.Laag; LabelColor = $colors.Text },
        @{ Name = 'Hoog'; Value = $row.Hoog; Color = $colors.Hoog; LabelColor = $colors.White },
        @{ Name = 'HeelHoog'; Value = $row.HeelHoog; Color = $colors.HeelHoog; LabelColor = $colors.White }
    )) {
        $w = ($segment.Value / 100) * $plotWidth
        $brush = New-Object System.Drawing.SolidBrush $segment.Color
        $g.FillRectangle($brush, [float]$x, [float]$y, [float]$w, [float]$barHeight)
        $brush.Dispose()

        if ($segment.Value -ge 7) {
            $label = "$($segment.Value)"
            $labelBrush = New-Object System.Drawing.SolidBrush $segment.LabelColor
            $size = $g.MeasureString($label, $fontTiny)
            $g.DrawString($label, $fontTiny, $labelBrush, [float]($x + $w / 2 - $size.Width / 2), [float]($y + ($barHeight - $size.Height) / 2))
            $labelBrush.Dispose()
        }
        $x += $w
    }

    if ($row.Land -eq 'Nederland') {
        $penHighlight = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml('#222222')), 2
        $g.DrawRectangle($penHighlight, [float]$left, [float]$y, [float](($row.Totaal / 100) * $plotWidth), [float]$barHeight)
        $penHighlight.Dispose()
    }

    $total = "$($row.Totaal)%"
    $g.DrawString($total, $fontSmall, $brushText, [float]($left + ($row.Totaal / 100) * $plotWidth + 8), [float]($y + ($barHeight - 15) / 2))
}

$legendY = $height - 88
$legendX = 35
foreach ($item in @(
    @{ Label = 'Laag'; Color = $colors.Laag },
    @{ Label = 'Hoog'; Color = $colors.Hoog },
    @{ Label = 'Heel hoog'; Color = $colors.HeelHoog }
)) {
    $brush = New-Object System.Drawing.SolidBrush $item.Color
    $g.FillRectangle($brush, [float]$legendX, [float]$legendY, 18, 18)
    $brush.Dispose()
    $g.DrawString($item.Label, $fontSmall, $brushText, [float]($legendX + 25), [float]($legendY - 1))
    $legendX += 130
}

$g.DrawString('Bron: CBS, Eurostat; CBS nieuwsbericht 30-03-2026.', $fontSmall, $brushMuted, 35, $height - 45)

$bmp.Save($outPath, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose()
$bmp.Dispose()

Write-Output "saved $outPath"
