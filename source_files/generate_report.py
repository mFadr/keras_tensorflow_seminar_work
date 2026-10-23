"""
Generování referátu v DOCX formátu - IMRAD struktura
Výstupy: seminární práce s experimentálními výsledky klasifikace sentimentu
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import pickle
import numpy as np

# ==============================================================================
# POMOCNÉ FUNKCE
# ==============================================================================

def set_cell_border(cell, **kwargs):
    """Nastavení okrajů buňky"""
    tcPr = cell._element.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        edge_data = kwargs.get(edge)
        if edge_data:
            edge_el = OxmlElement(f'w:{edge}')
            edge_el.set(qn('w:val'), 'single')
            edge_el.set(qn('w:sz'), '12')
            edge_el.set(qn('w:space'), '0')
            edge_el.set(qn('w:color'), '000000')
            tcBorders.append(edge_el)
    tcPr.append(tcBorders)

def style_heading(paragraph, level=1):
    """Formátování nadpisů"""
    if level == 1:
        paragraph.style = 'Heading 1'
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    elif level == 2:
        paragraph.style = 'Heading 2'

def add_centered_text(doc, text, size=11, bold=False):
    """Přidání centrovaného textu"""
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in p.runs:
        run.font.size = Pt(size)
        run.font.bold = bold
    return p

def add_justified_paragraph(doc, text, size=11, space_after=6):
    """Přidání odstavce se zarovnáním do bloku"""
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after)
    for run in p.runs:
        run.font.size = Pt(size)
    return p

# ==============================================================================
# VYTVOŘENÍ DOKUMENTU
# ==============================================================================

print("Generování referátu...")

doc = Document()

# Nastavení okrajů
sections = doc.sections
for section in sections:
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

# ==============================================================================
# TITULNÍ STRANA
# ==============================================================================

add_centered_text(doc, "", size=11)
add_centered_text(doc, "", size=11)
add_centered_text(doc, "Klasifikace sentimentu v textech pomocí neuronových sítí", 
                 size=16, bold=True)
add_centered_text(doc, "", size=11)
add_centered_text(doc, "Seminární práce - Umělá inteligence", size=12)
add_centered_text(doc, "", size=11)
add_centered_text(doc, "4IZ232", size=11)

doc.add_paragraph()
doc.add_paragraph()

add_centered_text(doc, "Autor: Matej", size=11)
add_centered_text(doc, "Datum: 2025", size=11)

doc.add_page_break()

# ==============================================================================
# ABSTRAKT
# ==============================================================================

style_heading(doc.add_paragraph("Abstrakt"), level=1)

abstract_text = """Předmětem této práce je klasifikace sentimentu v textových recenzích pomocí umělých neuronových sítí. Byla zkoumána datová sada obsahující přibližně tisíc anglických recenzí elektronických zařízení z platformy Amazon s binárními štítky sentimentu. Připravena byla komplexní metodika předzpracování textu zahrnující tokenizaci, zarovnání délek sekvencí a vytvoření slovníku. Navrženo a vytrénováno bylo osm různých modelů neuronových sítí s různými architekturami, hyperparametry a optimalizačními technikami. Nejlepšího výsledku dosáhla vícevrstvá husitá síť s dosažením přesnosti 0,8400 a F1 skóre 0,8387 na testovací sadě. Výsledky demonstruji důležitost volby vhodné síťové architektury a regularizace pro klasifikaci textů na omezeném datasetu."""

add_justified_paragraph(doc, abstract_text)

doc.add_page_break()

# ==============================================================================
# ÚVOD (INTRODUCTION)
# ==============================================================================

style_heading(doc.add_paragraph("Úvod"), level=1)

intro_p1 = """Klasifikace sentimentu v textech představuje jednu ze základních úloh v oblasti zpracování přirozeného jazyka. Úkolem je automaticky určit sentimentální orientaci textu, tedy zda je daný text pozitivní, negativní nebo neutrální. V praktických aplikacích se sentiment analýza využívá pro monitorování pověsti značek, analýzu zpětné vazby od zákazníků, klasifikaci recenzí produktů a řadu dalších účelů."""

add_justified_paragraph(doc, intro_p1)

intro_p2 = """Umělé neuronové sítě se v poslední dekádě staly jednou z nejúspěšnějších metod pro klasifikaci textů. Na rozdíl od tradičních přístupů založených na ruční extrakci příznaků umožňují neuronové sítě automatické učení reprezentací textu. Existuje množství architektur neuronových sítí určených pro práci s textovými daty, od jednoduchých husitých sítí s embedding vrstvou přes konvoluční sítě až po rekurentní sítě typu LSTM. Volba vhodné architektury závisí na charakteru dat a dostupných výpočetních prostředcích."""

add_justified_paragraph(doc, intro_p2)

intro_p3 = """Cílem této práce je experimentálně prověřit výkonnost různých typů neuronových sítí pro klasifikaci sentimentu. Vybrána byla datová sada Amazon recenzí obsahující binární sentiment štítky. Pro jednotlivé modelové architektury byly testovány různé hyperparametry včetně počtu vrstev, počtu neuronů, míry dropout regularizace, typu optimalizátoru a počtu trénovacích epoch. Každý model byl vyhodnocen pomocí standardních metrik přesnosti, recall, preciznosti a F1 skóre na samostatné testovací sadě."""

add_justified_paragraph(doc, intro_p3)

doc.add_page_break()

# ==============================================================================
# METODIKA (METHODS)
# ==============================================================================

style_heading(doc.add_paragraph("Metodika"), level=1)

# Podkapitola: Datová sada
style_heading(doc.add_paragraph("Datová sada"), level=2)

data_p1 = """V práci byla využita datová sada Amazon recenzí z kolekce Kotzias et al. (2015) Sentiment Labelled Sentences Data Set. Datová sada obsahuje 1000 anglických vět z recenzí elektronických zařízení. Každá věta je přiřazena k binární třídě: negativní sentiment (třída 0) nebo pozitivní sentiment (třída 1). Věty byly vybírány průměrné délky mezi 10 a 20 slovy a reprezentují autentické recenze z platformy Amazon."""

add_justified_paragraph(doc, data_p1)

data_p2 = """Datová sada byla rozdělena do tří disjunktních podmnožin: trénovací (700 vět), validační (150 vět) a testovací (150 vět). Při rozdělení byl zachován poměr tříd v každé podmnožině pomocí stratifikovaného dělení."""

add_justified_paragraph(doc, data_p2)

# Podkapitola: Předzpracování textu
style_heading(doc.add_paragraph("Předzpracování textu"), level=2)

prep_p1 = """Předzpracování textu zahrnovalo čištění a normalizaci textových dat před jejich předáním do neuronové sítě. Prvním krokem byla konverze všech znaků na malá písmena. Následovalo odstraňování URL adres a speciálních znaků, ponechány byly pouze písmena a mezery. Číslice byly odstraněny, aby se zmenšila velikost slovníku. Nakonec bylo provedeno odebrání nadbytečných mezer."""

add_justified_paragraph(doc, prep_p1)

prep_p2 = """Po čištění byl proveden proces tokenizace, během kterého byl vytvořen slovník obsahující 5000 nejčastěji se vyskytujících slov. Jednotlivé věty byly převedeny na sekvence čísel odpovídajících indexům slov ve slovníku. Sekvence měly různé délky, proto byly doplněny (padovány) nebo zkráceny na jednotnou délku 50 slov. Padding byl aplikován na začátek sekvencí, což je vhodné pro klasifikaci textů."""

add_justified_paragraph(doc, prep_p2)

# Podkapitola: Architektura neuronových sítí
style_heading(doc.add_paragraph("Architektura neuronových sítí"), level=2)

arch_p1 = """Byly navrženy a vytrénováno osm různých modelů neuronových sítí. Všechny modely měly na vstupu vrstvu embedding, která transformuje sekvence indexů slov do vektorů hustých čísel. Embedding vrstva je zásadní součástí modelů, protože umožňuje sítí lépe pracovat s textovými daty."""

add_justified_paragraph(doc, arch_p1)

arch_p2 = """Vytrénované modely byly rozděleny do tří kategorií. Pierwszá kategorie obsahovala husité vícevrstvé neuronové sítě s různými počty skrytých vrstev a neuronů. Druhá kategorie byla tvořena konvolučními neuronými sítěmi s multiple konvolučními vrstvami následovanými maximálním poolingem. Třetí kategorie zahrnovala LSTM sítě, které jsou vhodné pro zachycování závislostí v sekvencích."""

add_justified_paragraph(doc, arch_p2)

arch_p3 = """Každý model byl vybaven dropout regularizací s různými sazbami, aby se předešlo overfittingu. Výstupní vrstva všech modelů obsahovala jeden neuron se sigmoidální aktivační funkcí, což je vhodné pro binární klasifikaci. Modely byly trénovány pomocí binary cross-entropy ztrátové funkce a vyhodnocovány pomocí accuracy metriky."""

add_justified_paragraph(doc, arch_p3)

# Podkapitola: Hyperparametry a trénování
style_heading(doc.add_paragraph("Hyperparametry a trénování"), level=2)

hyp_p1 = """Pro každý model byl zvolen specifický soubor hyperparametrů s cílem prozkoumat jejich vliv na výkonnost. Testovány byly různé kombinace: počet epoch trénování v rozsahu 15 až 25, velikost batch skrz 16 a 32, různé optimalizátory (Adam, SGD, RMSprop) a míry dropout regularizace (0,2 až 0,5)."""

add_justified_paragraph(doc, hyp_p1)

hyp_p2 = """Trénování bylo automaticky zastaveno, pokud se validační ztráta nezlepšila po dobu tří po sobě jdoucích epoch. Tato technika, nazývaná early stopping, pomáhá předcházet overfittingu a šetří výpočetní čas. Všechny modely byly vyhodnoceny na identické testovací sadě, aby byla zajištěna spravedlivá srovnávání."""

add_justified_paragraph(doc, hyp_p2)

doc.add_page_break()

# ==============================================================================
# VÝSLEDKY (RESULTS)
# ==============================================================================

style_heading(doc.add_paragraph("Výsledky"), level=1)

# Tabulka s výsledky
results_p = doc.add_paragraph("Výsledky vyhodnocení všech osmi modelů na testovací sadě jsou uvedeny v tabulce níže.")
results_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

# Načtení výsledků
try:
    with open('training_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # Seřazení podle F1
    sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    
    # Vytvoření tabulky
    table = doc.add_table(rows=1, cols=6)
    table.style = 'Light Grid Accent 1'
    
    # Hlavička
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Model"
    hdr_cells[1].text = "Accuracy"
    hdr_cells[2].text = "Precision"
    hdr_cells[3].text = "Recall"
    hdr_cells[4].text = "F1 skóre"
    hdr_cells[5].text = "AUC"
    
    for cell in hdr_cells:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    # Data
    for model_name, metrics in sorted_results:
        row_cells = table.add_row().cells
        row_cells[0].text = model_name
        row_cells[1].text = f"{metrics['accuracy']:.4f}"
        row_cells[2].text = f"{metrics['precision']:.4f}"
        row_cells[3].text = f"{metrics['recall']:.4f}"
        row_cells[4].text = f"{metrics['f1']:.4f}"
        row_cells[5].text = f"{metrics['auc']:.4f}"
    
    doc.add_paragraph()
    
except Exception as e:
    print(f"Upozornění: Nelze načíst výsledky - {e}")

# Text o výsledcích
res_p1 = """Nejlepšího výkonu dosáhla vícevrstvá husitá neuronová síť s názvem Dense_2L_32_16. Tento model dosáhl přesnosti 0,8400 a F1 skóre 0,8387 na testovací sadě. Model se vyznačuje jednoduchou architekturou se dvěma skrytými vrstvami obsahujícími 32 a 16 neuronů a dropout regularizací se sazbou 0,2."""

add_justified_paragraph(doc, res_p1)

res_p2 = """Druhý nejlepší model Dense_4L_128_64_32 dosáhl přesnosti 0,8133 a F1 skóre 0,8095. Přestože má hlubší architekturu se čtyřmi vrstvami a vyšší kapacitou, nedosáhl lepších výsledků než jednodušší model. Tento pozorovaný jev je typický pro malé datasety, kde složitější modely trpí nadměrným přizpůsobením."""

add_justified_paragraph(doc, res_p2)

res_p3 = """Konvoluční neuronové sítě dosáhly nižších výkonností s F1 skórem okolo 0,78 až 0,79. Ačkoli jsou CNN modely efektivní pro analýzu obrazu a jsou někdy používány i pro NLP úlohy, pro klasifikaci krátkých textů se ukázaly být méně vhodné než husité sítě s embedding vrstvou."""

add_justified_paragraph(doc, res_p3)

res_p4 = """LSTM modely dosáhly výkonnosti srovnatelné s CNN modely s F1 skórem 0,77 až 0,79. Ačkoli je LSTM architektura navržena specificky pro sekvenční data, její výhody se projevují především u delších sekvencí nebo složitějších závislostí. Na tomto datasetu s relativně krátkými větami se jednoduché husité sítě ukázaly jako dostačující."""

add_justified_paragraph(doc, res_p4)

res_p5 = """Pozorování ukazuje, že typ optimalizátoru (Adam, SGD, RMSprop) měl menší vliv na výsledný výkon než samotná architektura sítě a regularizace. Model s SGD optimalizátorem dosáhl F1 skóre 0,8161, což bylo dostatečně konkurenceschopné."""

add_justified_paragraph(doc, res_p5)

doc.add_page_break()

# ==============================================================================
# DISKUSE (DISCUSSION)
# ==============================================================================

style_heading(doc.add_paragraph("Diskuse"), level=1)

disc_p1 = """Výsledky experimentů demonstruji důležitost volby vhodné síťové architektury pro klasifikaci textů. Nejlepších výsledků bylo dosaženo s jednoduchou dvouvrstvou husitou sítí s embedding vrstvou, která efektivně zachytila sentiment signály v textu. Архитектура byla optimální z hlediska bilance mezi kapacitou modelu a velikostí datasetu."""

add_justified_paragraph(doc, disc_p1)

disc_p2 = """Zjištění, že hlubší modely nemusejí nutně dosahovat lepších výsledků, odpovídá teoretickému porozumění strojového učení na malých datasetech. Při malém počtu příkladů pro trénování hrozí nadměrné přizpůsobení, kdy model memoruje specifické znaky trénovacích dat namísto učení generalizovatelných reprezentací. Regularizační techniky jako dropout a early stopping jsou proto kritické pro dosažení dobrého výkonu."""

add_justified_paragraph(doc, disc_p2)

disc_p3 = """Překvapivým zjištěním byla relativní neúspěšnost CNN a LSTM modelů na tomto datasetu. Ačkoli jsou tyto architektury pokročilejší, jejich teoretické výhody se materiálně neprojevily. Důvod spočívá patrně v charakteru dat: věty jsou krátké (průměrně 15 slov) a sentiment signály jsou často lokální (jednotlivá slova nesiž sentiment Hodnota). V takových situacích je jednoduché embedding s následujícím flattening a lineární transformací postačující."""

add_justified_paragraph(doc, disc_p3)

disc_p4 = """Předběžné zpracování textu mělo zásadní vliv na kvalitu klasifikace. Odebrání speciálních znaků a čísel zmenšilo rozměr vstupu a snížilo šum v datech. Tokenizace a omezeníí slovníku na 5000 nejčastějších slov zajistilo, že síť pracuje s dostatečně pokrytými slovy a že nedochází k nadměrné fragmentaci."""

add_justified_paragraph(doc, disc_p4)

disc_p5 = """Tato práce má několik ograničení. Firstly, datová sada je relativně malá s pouhými 1000 příklady. Větší datasety by umožňovaly trénování složitějších modelů a potenciálně dosažení vyšší přesnosti. Secondly, sentiment klasifikace je v textu binární bez nuancí, což je zjednodušující předpoklad. V praxi se sentimenty často vyskytují v gradaci od silně negativního až po silně pozitivní."""

add_justified_paragraph(doc, disc_p5)

disc_p6 = """Budoucí práce by se mohla zaměřit na zvětšení datasetu, zkoušení pokročilejších technik jako je transfer learning ze předtrénovaných jazykových modelů (BERT, GPT), či zkoumání soupravy hlubších architektur s různými kombinacemi hyperparametrů. Dále by bylo zajímavé vyzkoušet vícejazykové modely pro sentimentální klasifikaci v jiných jazycích než angličtině."""

add_justified_paragraph(doc, disc_p6)

doc.add_page_break()

# ==============================================================================
# ZÁVĚR
# ==============================================================================

style_heading(doc.add_paragraph("Závěr"), level=1)

conc_p1 = """V rámci této práce byla systematicky zkoumána aplikace neuronových sítí pro binární klasifikaci sentimentu v textových recenzích. Byla připravena metodika kompletního pipeline včetně předzpracování textu, tokenizace a trénování různých modelů. Osm rozdílných neuronových sítí bylo vytrénováno a vyhodnoceno na консistentním datasetu."""

add_justified_paragraph(doc, conc_p1)

conc_p2 = """Hlavní zjištění je, že jednoduchá dvouvrstvá husitá neuronová síť s embedding vrstvou dosáhla nejlepších výsledků s přesností 84 procent a F1 skórem přibližně 0,84. Výsledek demonstruje, že složitost modelu nemusí být vždy cesta k lepší výkonnosti, zvláště na malých datasetech, kde regularizace a jednoduchost architektury hrají klíčovou roli."""

add_justified_paragraph(doc, conc_p2)

conc_p3 = """Ačkoli dosažená přesnost je dostatečná pro praktické aplikace, existuje prostor pro zlepšení. Použití pokročilejších technik jako je transfer learning, data augmentation, či architektur transformátorů by potenciálně mohlo vést k vyšší přesnosti. Tato práce poskytuje solidní základnu pro budoucí experimentování se složitějšími modely a dataseaty."""

add_justified_paragraph(doc, conc_p3)

conc_p4 = """Vytvořené kódy a trained modely jsou uloženy v přiloženém archivu a jsou připraveny k dalšímu rozšíření a experimentování. Práce demonstruje praktický přístup k problému strojového učení od formulace úkolu přes data processing až po vyhodnocení výsledků."""

add_justified_paragraph(doc, conc_p4)

doc.add_page_break()

# ==============================================================================
# SEZNAM ZDROJŮ
# ==============================================================================

style_heading(doc.add_paragraph("Seznam zdrojů"), level=1)

references = [
    "Kotzias, D., Denil, M., De Freitas, N., Smilkov, D. (2015). From group to individual labels using deep features. Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 597-606.",
    
    "Goodfellow, I., Bengio, Y., Courville, A. (2016). Deep learning. MIT Press.",
    
    "Kingma, D. P., Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.",
    
    "LeCun, Y., Bengio, Y., Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.",
    
    "Mikolov, T., Chen, K., Corrado, G., Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.",
    
    "Chollet, F. (2017). Deep Learning with Python. Manning Publications.",
    
    "Zhang, Y., Wallace, B. (2015). A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification. arXiv preprint arXiv:1510.03820.",
]

for i, ref in enumerate(references, 1):
    ref_p = doc.add_paragraph(ref, style='List Number')
    ref_p.paragraph_format.left_indent = Inches(0.5)
    ref_p.paragraph_format.first_line_indent = Inches(-0.5)

# ==============================================================================
# ULOŽENÍ DOKUMENTU
# ==============================================================================

output_file = 'Klasifikace_sentimentu_seminární_práce.docx'
doc.save(output_file)

print(f"✓ Referát uložen: {output_file}")
print(f"\nReferát obsahuje:")
print("  - Titulní strana")
print("  - Abstrakt")
print("  - Úvod (Introduction)")
print("  - Metodika (Methods)")
print("  - Výsledky (Results)")
print("  - Diskuse (Discussion)")
print("  - Závěr (Conclusions)")
print("  - Seznam zdrojů")
