from constants import RESULTS_PATH

import re

def fix_decimal_commas(text):
    # Replace commas that are between digits with a dot
    # e.g. "123,456" → "123.456"
    return re.sub(r'(?<=\d),(?=\d)', '.', text)

file = f"{RESULTS_PATH}/satscan_utah/results.clustermap.html"

# Example usage
with open(file, "r", encoding="utf-8") as f:
    html = f.read()

fixed_html = fix_decimal_commas(html)

output_file = f"{RESULTS_PATH}/satscan_utah/results_fixed.clustermap.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(fixed_html)

print("Fixed HTML saved to Map_fixed.html")
