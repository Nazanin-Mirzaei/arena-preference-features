import re
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

# 1. SETUP
TARGET_LANGUAGES = [
    "python", "java", "c", "cpp", "csharp", "go", "rust", "swift", "kotlin",
    "javascript", "typescript", "php", "ruby", "dart", "graphql", "html", "css",
    "bash", "powershell", "perl", "lua", "r", "bat",
    "sql", "scala", "julia", "docker", "makefile", "terraform", "nginx",
    "json", "yaml", "toml", "xml", "ini", "markdown", "diff"
]

ALLOWED_LEXERS = []
for lang in TARGET_LANGUAGES:
    try:
        lexer_inst = get_lexer_by_name(lang)
        ALLOWED_LEXERS.append(type(lexer_inst))
    except: continue

# 2. PATTERNS (Expanded for Fragments)
LANGUAGE_PATTERNS = {
    # High Priority (Distinct Syntax)
    "HTML": [r"<!DOCTYPE\s+html>", r"<\w+>.*</\w+>", r"<div\s+class=", r"<span", r"<br/?>"],
    "CSS": [r"^\s*[.#]\w+\s*\{[\s\S]*?:\s*[\s\S]*?;", r"@media\s+screen", r"\.[\w-]+\s*\{"], 
    "Diff": [r"^diff\s+--git", r"^\s*-\s+.*\n\s*\+\s+.*", r"^\s*@@\s+-\d+"],
    "SQL": [r"\b(SELECT|INSERT\s+INTO|UPDATE|DELETE|DROP|CREATE\s+TABLE|ALTER\s+TABLE)\b", r"VALUES\s*\("],
    "JSON": [r"^\s*\{\s*\"", r"^\s*\"[\w]+\":\s*[\w\d\"]+"], # Catch fragments "key": val
    
    # Systems
    "Python": [r"def\s+\w+\(.*\):", r"import\s+[\w.]+", r"class\s+\w+\(.*\):", r"\[.*for\s+\w+\s+in\s+.*\]", r"^\w+\s*=\s*[\w\[\{]"],
    "Java": [r"public\s+class\s+\w+", r"public\s+static\s+void\s+main", r"public\s+\w+\s+\w+\(.*\)", r"void\s+\w+\(.*\)\s*\{"],
    "C": [r"#include\s*<stdio\.h>", r"int\s+main\s*\(.*\)\s*\{"],
    "Cpp": [r"#include\s*<iostream>", r"using\s+namespace\s+std;"],
    "JavaScript": [r"const\s+\w+\s*=", r"function\s+\w+\(.*\)\s*\{", r"console\.log", r"return\s+(true|false)", r"=>", r"\{\s*\w+:\s*\""],
    
    # Shell/DevOps
    "Bash": [r"^#!\/bin\/bash", r"echo\s+['\"].*['\"]", r"sudo\s+\w+", r"(npm|pip|git|docker|make)\s+\w+"],
    "Docker": [r"^FROM\s+\w+", r"^RUN\s+\w+", r"^COPY\s+"],
    "Terraform": [r"resource\s+\"[\w_]+\"", r"provider\s+\"[\w_]+\""],
    "Nginx": [r"location\s+.*\s*\{", r"server\s*\{"],
    "Makefile": [r"^\w+:\s*\n\t+"],

    # Modern Langs
    "Go": [r"package\s+main", r"func\s+main", r"type\s+\w+\s+struct"],
    "Rust": [r"fn\s+main", r"struct\s+\w+\s*\{", r"impl\s+\w+"],
    "Kotlin": [r"fun\s+main", r"data\s+class\s+\w+"],
    "Scala": [r"case\s+class\s+\w+"],
    "Swift": [r"var\s+body:\s+some\s+View", r"Text\(.*\)"],
    "Ruby": [r"\d+\.times\s*\{", r"def\s+\w+"],
    "PHP": [r"<\?php", r"<\?="],

    # Config (Lowest Priority)
    "YAML": [r"^\s*-\s+\w+:", r"^\s*\w+:\s*['\"].*['\"]", r"^\s*-\s+\w+$" , r"^\s*[\w\-\.]+\s*:\s*[^\{\};]+$"], 
    "TOML": [r"\[.*\]", r"^\w+\s*=\s*.*"],
}

# 3. KEYWORDS
LANGUAGE_KEYWORDS = {
    "Python": ["print(", "None", "True", "False"],
    "JavaScript": ["undefined", "null", "NaN", "document"],
    "SQL": ["SELECT", "FROM", "WHERE", "JOIN", "VALUES"],
    "Bash": ["ls", "cd", "mkdir", "rm", "cp", "npm", "pip", "git", "docker"],
    "JSON": ["true", "false", "null"],
    "CSS": ["margin", "padding", "color", "background", "border"],
}

def is_likely_text(content):
    lines = content.strip().split('\n')
    if not lines: return False
    list_markers = ('-', '*', '1.', 'Step ', 'Log:', 'Error:', '[INFO]')
    marker_count = sum(1 for line in lines if line.strip().startswith(list_markers))
    return (marker_count / len(lines) > 0.5)

def detect_code(text):
    pattern = r"```([\w\+#\-]+)?\n([\s\S]*?)```"
    matches = re.finditer(pattern, text, re.DOTALL)
    results = []
    
    for match in matches:
        lang_tag = match.group(1)
        content = match.group(2).strip()
        
        # 1. Explicit
        if lang_tag:
            results.append(f"[Explicit] {lang_tag}")
            continue

        # 2. Pygments
        best_lang = "Plain Text"
        best_score = 0.0
        for lexer_cls in ALLOWED_LEXERS:
            try:
                score = lexer_cls.analyse_text(content)
                if score > best_score:
                    best_score = score
                    best_lang = lexer_cls.name
            except: continue
        
        if best_score > 0.1:
            results.append(f"PASS: {best_lang} (Pygments)")
            continue

        # 3. Regex
        regex_hit = None
        for lang, patterns in LANGUAGE_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, content, re.MULTILINE | re.IGNORECASE):
                    regex_hit = lang
                    break
            if regex_hit: break
        
        if regex_hit:
            results.append(f"PASS: {regex_hit} (Regex)")
            continue

        # 4. Keywords
        keyword_hit = None
        for lang, keywords in LANGUAGE_KEYWORDS.items():
            if sum(1 for k in keywords if k in content) >= 2:
                keyword_hit = lang
                break
        if keyword_hit:
            results.append(f"PASS: {keyword_hit} (Keywords)")
            continue

        results.append("FAIL: Plain Text (Unknown)")

    return results
