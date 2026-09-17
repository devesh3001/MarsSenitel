import zipfile
import xml.etree.ElementTree as ET

def read_docx(path):
    with zipfile.ZipFile(path) as z:
        xml_content = z.read('word/document.xml')
        tree = ET.fromstring(xml_content)
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        return '\n'.join([node.text for node in tree.findall('.//w:t', ns) if node.text])

print('=== DA PS ===')
print(read_docx("DA PS 26'.docx"))
print('\n=== RULES AND REGULATIONS ===')
print(read_docx("RULES AND REGULATIONS.docx"))
print('\n=== DATA DESCRIPTION ===')
print(read_docx("DATA_DESCRIPTION.docx"))
