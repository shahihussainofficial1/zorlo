#pragma once
// Turns a file into text the (text-only) model can read, with only what Windows has built in:
//   Word / PowerPoint / Excel (.docx .pptx .xlsx, also .odt .ods .odp): the text from their XML inside the zip
//   PDF: the text in the file; for scanned PDFs (no text) the pages are rendered and read with Windows OCR
//   pictures: Windows OCR (the words in screenshots, photos of documents) plus the size
//   video / audio: length and size (the model can't watch or listen)
// Run by agent.cpp with $Path, $Kind and $Out set; writes {kind, text, pages, width, height, duration_s, note} to $Out.

static const char * EXTRACT_PS = R"PS(
$ErrorActionPreference = 'Stop'
$MaxText = 400000
$r = [ordered]@{ kind = $Kind; text = ''; note = '' }
function Clip($t) { if ($t.Length -gt $MaxText) { $t.Substring(0, $MaxText) + "`n[... cut]" } else { $t } }
function XmlText($x) {
  if (-not $x) { return '' }
  $x = $x -replace '</w:p>|</a:p>|</text:p>|</text:h>|<w:br[^>]*/>|<text:line-break/>', "`n" -replace '<w:tab/>|<text:tab/>', "`t"
  $x = $x -replace '<[^>]+>', ''
  ([Net.WebUtility]::HtmlDecode($x) -replace "`n{3,}", "`n`n").Trim()
}
function Entry($z, $n) { $e = $z.GetEntry($n); if (-not $e) { return $null }; $s = New-Object IO.StreamReader($e.Open(), [Text.Encoding]::UTF8); try { $s.ReadToEnd() } finally { $s.Close() } }
function Num($name) { [int](([regex]::Match($name, '(\d+)\.xml$')).Groups[1].Value) }

# WinRT from Windows PowerShell 5.1: OCR, PDF rendering, image decoding
function WinRT {
  Add-Type -AssemblyName System.Runtime.WindowsRuntime
  $script:asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
  $script:asAction = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction' } | Select-Object -First 1
  $null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
  $null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
  $null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]
  $null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
  $null = [Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType = WindowsRuntime]
  $script:ocr = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
}
function Await($op, [Type]$t) { $task = $script:asTask.MakeGenericMethod($t).Invoke($null, @($op)); $null = $task.Wait(-1); $task.Result }
function AwaitAction($op) { $null = $script:asAction.Invoke($null, @($op)).Wait(-1) }
function OcrStream($stream) {
  $dec = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
  $script:w = $dec.PixelWidth; $script:h = $dec.PixelHeight
  if (-not $script:ocr) { return $null }
  $bmp = Await ($dec.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
  $max = [Windows.Media.Ocr.OcrEngine]::MaxImageDimension
  if ($dec.PixelWidth -gt $max -or $dec.PixelHeight -gt $max) {
    $scale = [Math]::Min($max / $dec.PixelWidth, $max / $dec.PixelHeight)
    $tf = New-Object Windows.Graphics.Imaging.BitmapTransform
    $tf.ScaledWidth = [uint32]($dec.PixelWidth * $scale); $tf.ScaledHeight = [uint32]($dec.PixelHeight * $scale)
    $bmp = Await ($dec.GetSoftwareBitmapAsync([Windows.Graphics.Imaging.BitmapPixelFormat]::Bgra8, [Windows.Graphics.Imaging.BitmapAlphaMode]::Premultiplied, $tf,
      [Windows.Graphics.Imaging.ExifOrientationMode]::RespectExifOrientation, [Windows.Graphics.Imaging.ColorManagementMode]::DoNotColorManage)) ([Windows.Graphics.Imaging.SoftwareBitmap])
  }
  $res = Await ($script:ocr.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult])
  ($res.Lines | ForEach-Object { $_.Text }) -join "`n"
}
function OpenFile($p) { Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($p)) ([Windows.Storage.StorageFile]) }

# text drawn by a PDF's content streams: literal (strings) of Tj/TJ, which covers most PDFs made from documents
function PdfText($bytes) {
  $raw = [Text.Encoding]::GetEncoding(28591).GetString($bytes)
  $sb = New-Object Text.StringBuilder
  foreach ($m in [regex]::Matches($raw, '(?s)obj\s*<<((?:(?!endobj).){0,3000}?)>>\s*stream\r?\n')) {
    $start = $m.Index + $m.Length
    $end = $raw.IndexOf('endstream', $start)
    if ($end -lt 0) { continue }
    $data = $null
    if ($m.Groups[1].Value -match '/FlateDecode') {
      try {
        # ::new, not New-Object: New-Object spreads a byte[] argument into separate arguments
        $ms = [IO.MemoryStream]::new($bytes, $start + 2, [Math]::Max(0, $end - $start - 2))
        $ds = [IO.Compression.DeflateStream]::new($ms, [IO.Compression.CompressionMode]::Decompress)
        $outm = [IO.MemoryStream]::new(); $ds.CopyTo($outm)
        $data = [Text.Encoding]::GetEncoding(28591).GetString($outm.ToArray())
      } catch { continue }
    } elseif ($m.Groups[1].Value -notmatch '/Filter') { $data = $raw.Substring($start, $end - $start) }
    if (-not $data -or $data -notmatch 'BT') { continue }
    foreach ($bt in [regex]::Matches($data, '(?s)BT(.*?)ET')) {
      $line = New-Object Text.StringBuilder
      foreach ($t in [regex]::Matches($bt.Groups[1].Value, '\((?:\\.|[^\\)])*\)|T\*|Td|TD|''')) {
        $v = $t.Value
        if ($v -in 'T*', 'Td', 'TD', "'") { if ($line.Length -and $line[$line.Length - 1] -ne "`n") { [void]$line.Append("`n") }; continue }
        $s = $v.Substring(1, $v.Length - 2) -replace '\\n', "`n" -replace '\\r', '' -replace '\\t', "`t"
        $s = [regex]::Replace($s, '\\([0-7]{1,3})', { param($o) [string][char][Convert]::ToInt32($o.Groups[1].Value, 8) })
        $s = $s -replace '\\(.)', '$1'
        [void]$line.Append($s)
      }
      if ($line.Length) { [void]$sb.AppendLine(($line.ToString() -replace '[ \t]+', ' ' -replace ' ?\n ?', "`n").Trim()) }
    }
  }
  $sb.ToString()
}

try {
  switch ($Kind) {
    { $_ -in 'docx', 'pptx', 'xlsx', 'odt', 'ods', 'odp' } {
      Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
      $z = [IO.Compression.ZipFile]::OpenRead($Path)
      try {
        if ($Kind -eq 'docx') { $r.text = XmlText ((Entry $z 'word/document.xml') + (($z.Entries | Where-Object { $_.FullName -match '^word/(footnotes|endnotes)\.xml$' } | ForEach-Object { Entry $z $_.FullName }) -join '')) }
        elseif ($Kind -eq 'pptx') {
          $slides = $z.Entries | Where-Object { $_.FullName -match '^ppt/slides/slide\d+\.xml$' } | Sort-Object { Num $_.FullName }
          $r.pages = @($slides).Count
          $r.text = ($slides | ForEach-Object { "--- Slide $(Num $_.FullName) ---`n" + (XmlText (Entry $z $_.FullName)) }) -join "`n`n"
        }
        elseif ($Kind -eq 'xlsx') {
          $ss = @()
          $sx = Entry $z 'xl/sharedStrings.xml'
          if ($sx) { $ss = @(([xml]$sx).sst.si | ForEach-Object { $_.InnerText }) }
          $names = @(([xml](Entry $z 'xl/workbook.xml')).workbook.sheets.sheet | ForEach-Object { $_.name })
          $sheets = $z.Entries | Where-Object { $_.FullName -match '^xl/worksheets/sheet\d+\.xml$' } | Sort-Object { Num $_.FullName }
          $sb = New-Object Text.StringBuilder; $i = 0
          foreach ($s in $sheets) {
            $nm = if ($i -lt $names.Count) { $names[$i] } else { "Sheet$($i + 1)" }; $i++
            [void]$sb.AppendLine("--- Sheet: $nm ---")
            $rows = 0
            foreach ($row in ([xml](Entry $z $s.FullName)).worksheet.sheetData.row) {
              if (++$rows -gt 5000) { [void]$sb.AppendLine('[... more rows]'); break }
              $cells = @{}; $maxc = 0
              foreach ($c in @($row.c)) {
                if (-not $c) { continue }
                $col = 0; foreach ($ch in ([regex]::Match([string]$c.r, '^[A-Z]+').Value).ToCharArray()) { $col = $col * 26 + ([int]$ch - 64) }
                $v = if ($c.t -eq 's') { $ss[[int]$c.v] } elseif ($c.t -eq 'inlineStr') { $c.is.InnerText } else { [string]$c.v }
                $cells[$col] = $v; if ($col -gt $maxc) { $maxc = $col }
              }
              [void]$sb.AppendLine((1..[Math]::Max(1, $maxc) | ForEach-Object { $cells[$_] }) -join "`t")
            }
          }
          $r.pages = @($sheets).Count
          $r.text = $sb.ToString()
        }
        else { $r.text = XmlText (Entry $z 'content.xml') }
      } finally { $z.Dispose() }
    }
    'pdf' {
      $bytes = [IO.File]::ReadAllBytes($Path)
      $t = PdfText $bytes
      $letters = ([regex]::Matches($t, '[\p{L}\p{N}]')).Count
      $pages = ([regex]::Matches([Text.Encoding]::GetEncoding(28591).GetString($bytes), '/Type\s*/Page[^s]')).Count
      if ($pages) { $r.pages = $pages }
      # its own text, unless that is (nearly) empty or garbled; then it's treated as scanned
      if ($letters -ge [Math]::Max(20, 40 * [Math]::Max(1, $pages) / 4) -and $letters / [Math]::Max(1, $t.Length) -gt 0.5) { $r.text = $t; $r.note = 'text taken from the PDF' }
      else {
        WinRT
        if (-not $script:ocr) { $r.text = $t; $r.note = 'this PDF has little readable text and Windows OCR is not available (Settings > Time & language > Language: add a language with OCR)'; break }
        $pdf = Await ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync((OpenFile $Path))) ([Windows.Data.Pdf.PdfDocument])
        $r.pages = $pdf.PageCount
        $n = [Math]::Min($pdf.PageCount, 40)
        $sb = New-Object Text.StringBuilder
        for ($i = 0; $i -lt $n; $i++) {
          $page = $pdf.GetPage($i)
          $ms = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
          $opt = New-Object Windows.Data.Pdf.PdfPageRenderOptions
          $opt.DestinationWidth = [uint32]1700
          AwaitAction ($page.RenderToStreamAsync($ms, $opt))
          [void]$sb.AppendLine("--- Page $($i + 1) ---"); [void]$sb.AppendLine((OcrStream $ms))
        }
        $r.text = $sb.ToString()
        $r.note = "scanned PDF: read with Windows OCR ($n of $($pdf.PageCount) pages); check names and numbers"
      }
    }
    'image' {
      WinRT
      $st = Await ((OpenFile $Path).OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
      $t = OcrStream $st
      $r.width = $script:w; $r.height = $script:h
      if ($null -eq $t) { $r.note = 'Windows OCR is not available, so the words in the picture could not be read' }
      elseif ($t.Trim()) { $r.text = $t; $r.note = 'words read from the picture with Windows OCR; the model cannot see the picture itself' }
      else { $r.note = 'no words found in the picture; the model cannot see pictures, so ask the user to describe it if it matters' }
    }
    'media' {
      $sh = New-Object -ComObject Shell.Application
      $item = $sh.Namespace((Split-Path -Parent $Path)).ParseName((Split-Path -Leaf $Path))
      $d = $item.ExtendedProperty('System.Media.Duration'); if ($d) { $r.duration_s = [Math]::Round([double]$d / 1e7, 1) }
      $fw = $item.ExtendedProperty('System.Video.FrameWidth'); if ($fw) { $r.width = $fw; $r.height = $item.ExtendedProperty('System.Video.FrameHeight') }
      $r.note = 'the model cannot watch video or hear audio; only its length and size are known'
    }
  }
} catch { $r.error = $_.Exception.Message }
$r.text = Clip ([string]$r.text)
[IO.File]::WriteAllText($Out, ($r | ConvertTo-Json -Compress -Depth 3), (New-Object Text.UTF8Encoding($false)))
)PS";
