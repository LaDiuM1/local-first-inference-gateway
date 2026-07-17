"""하나의 Markdown 원본을 공개 `/docs` HTML로 렌더링한다."""

from pathlib import Path

from markdown_it import MarkdownIt

_HTML_SHELL = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>openAt Inference API</title>
  <style>
    :root { color-scheme: light dark; font-family: system-ui, sans-serif; line-height: 1.6; }
    body { margin: 0 auto; max-width: 960px; padding: 2rem 1.25rem 5rem; }
    h1, h2, h3 { line-height: 1.25; margin-top: 2rem; }
    pre { overflow-x: auto; padding: 1rem; border-radius: .5rem; background: #172033; color: #f3f6fb; }
    code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
    :not(pre) > code { padding: .1rem .3rem; border-radius: .25rem; background: color-mix(in srgb, currentColor 10%, transparent); }
    a { color: #3977d3; }
    blockquote { margin-left: 0; padding-left: 1rem; border-left: .25rem solid #708090; }
  </style>
</head>
<body>
<main>
{content}
</main>
</body>
</html>
"""


class PublicDocsError(Exception):
    """공개 연동 문서 원본을 읽을 수 없다."""


def render_public_docs(path: Path) -> str:
    try:
        markdown = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise PublicDocsError("public API guide is unavailable") from error
    # 연동 가이드의 별칭·제한 표가 계약의 핵심이다 — CommonMark에는 표가 없어
    # 명시적으로 켠다.
    renderer = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable(
        "table"
    )
    return _HTML_SHELL.replace("{content}", renderer.render(markdown))
