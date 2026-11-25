import markdown

with open('FabMetrics_AI_Evaluation_Guide.md', 'r', encoding='utf-8') as f:
    md_text = f.read()

html_body = markdown.markdown(md_text)

html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FabMetrics AI Evaluation Guide</title>
<style>
    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px auto; max-width: 800px; color: #333; }}
    h1, h2, h3 {{ color: #2c3e50; }}
    h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
    h2 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; }}
    code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
    ul, ol {{ margin-top: 5px; }}
    li {{ margin-bottom: 8px; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

with open('FabMetrics_AI_Evaluation_Guide.html', 'w', encoding='utf-8') as f:
    f.write(html_content)
print("Success")
