from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def create_report(output_path='reports/AIIS_WH2_report.pdf', summary_text=''):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    c = canvas.Canvas(output_path, pagesize=letter)
    c.setFont('Helvetica-Bold', 16)
    c.drawString(72, 720, 'AIIS-WH2 - COVID-19 Analysis Report')
    c.setFont('Helvetica', 10)
    c.drawString(72, 700, 'CRISP-DM: Business Understanding → Data → Modeling → Evaluation → Deployment')
    text = c.beginText(72, 660)
    text.setFont('Helvetica', 10)
    for line in summary_text.split('\n'):
        text.textLine(line)
    c.drawText(text)
    c.showPage()
    c.save()
    return output_path
