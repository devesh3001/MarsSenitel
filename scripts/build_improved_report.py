"""Create the second-stage report from completed, measured improvement outputs."""
from pathlib import Path
import json,csv
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
ROOT=Path(__file__).resolve().parents[1]

def rows(path):
    with open(path,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    selection=json.loads((ROOT/'outputs/improved_selection.json').read_text())
    run=ROOT/'outputs'/selection['run']
    analysis=run/'mixture_three_sigma_trees2000'
    calibration=json.loads((analysis/'calibration.json').read_text())
    diagnostics=json.loads((analysis/'diagnostics.json').read_text())
    hypotheses=json.loads((ROOT/'outputs/improved_hypotheses.json').read_text())
    chosen=rows(analysis/'selected_for_interpretation.csv')
    experiments=rows(ROOT/'outputs/comparison/experiments.csv')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8))
    styles.add(ParagraphStyle(name='SmallBody',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=6))
    styles['Title'].alignment=0;styles['Title'].fontSize=24;styles['Title'].leading=28
    styles['Heading1'].fontSize=16;styles['Heading1'].leading=20
    story=[]
    def p(text,style='Body'):story.append(Paragraph(text,styles[style]))
    def h(text):p(text,'Heading1')
    def page():story.append(PageBreak())
    def picture(path,width=490):
        from reportlab.lib.utils import ImageReader
        w,h=ImageReader(str(path)).getSize()
        story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,8))
    def table(data,widths):
        t=Table([[Paragraph(escape(str(x)),styles['SmallBody']) for x in row] for row in data],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6edf3')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#cad3dc')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(t);story.append(Spacer(1,10))

    p('Mars image anomaly detection','Title')
    p('NSSC 2026 | My preprocessing experiments and final analysis','Heading2')
    if selection.get('status')!='Complete':p('<b>DRAFT: independent checks or verification are incomplete.</b>')
    p('I trained an autoencoder from scratch and used Isolation Forest on its latent vector. The first candidate list was dominated by unusually bright images and changed across repeated runs. I added controlled brightness and footprint experiments to investigate that weakness.')
    h('My decision')
    p(escape(selection['rationale']))
    p(f"<b>Reference run: {selection['run']}.</b> The calibrated boundary is {calibration['threshold']:.6f}; {diagnostics['flagged_total']} of 10,422 crops exceed it. The 95% source-bootstrap threshold interval is [{calibration['bootstrap_interval_95'][0]:.6f}, {calibration['bootstrap_interval_95'][1]:.6f}].")
    p('<b>What this establishes.</b> '+escape(selection['improvement_claim']))
    p('<b>What it does not establish.</b> Hidden anomaly labels are unavailable. I cannot claim precision, recall, F1 or a verified false-positive rate. Candidate appearance and reproducibility are not ground truth.')
    selected_runs={'v1','v2','v3','v4','v5','improved_seed','improved_split'}
    table([['Run','Input','Val. MSE','Val. SSIM','Flags']]+[[r['run'],r['input_domain'],f"{float(r['validation_mse']):.6f}",f"{float(r['validation_ssim']):.4f}",r.get('flagged','')] for r in experiments if r['run'] in selected_runs],[110,100,95,95,90])
    p('Metrics use each run\'s best validation-objective checkpoint. Different preprocessing changes the reconstruction target, and the split repeat uses different validation images. Their MSE/SSIM are not interchangeable quality scores. All runs use random initialization.','SmallBody')
    p('The notebook is prepared for Colab/Kaggle. These saved experiments ran on a local RTX 3050; no cloud execution is invented.','SmallBody')

    page();h('Data and the preprocessing comparison')
    p('I decoded all 10,422 supplied crops, verified 227 by 227 grayscale inputs, checked metadata joins and found no exact decoded-pixel duplicates. There are 172 source observations. Near-duplicate or geographically overlapping observations remain possible.')
    table([['Main split','Sources','Crops'],['Training','120','7,101'],['Validation','25','1,666'],['Calibration','27','1,655']],[240,100,150])
    p('I keep each source in one split. v4 changes only preprocessing: subtract the crop mean, divide by standard deviation, map with 0.5 + 0.15*z, then clip to [0,1]. Standard deviation is floored at 1/255. v5 adds exclusion of exact-zero pixels connected to the image border when estimating the transform, then fills them with 0.5. Disconnected interior zeros remain content.')
    picture(ROOT/'outputs/comparison/preprocessing_examples.png',width=290)
    p('Figure 1. Fixed validation examples: raw, contrast-standardized and footprint-treated inputs. The mask is a heuristic, not an organizer-supplied validity mask.','SmallBody')
    p('This intentionally suppresses global brightness and can remove meaningful albedo information. Clipping can suppress extreme contrast; normalization can amplify noise; a connected shadow can be mistaken for an image footprint.')

    page();h('Phase 1 Architecture, loss and learning curves')
    p('Five stride-two convolution blocks use channels 16, 32, 64, 96 and 128, with GroupNorm and SiLU. Spatial sizes are 114, 57, 29, 15 and 8. A linear layer creates a fixed 128- or 256-dimensional vector. The decoder expands the vector and uses resize-convolution to reconstruct exactly 227 by 227. There is no bypass connection and no pretrained feature extractor.')
    p('v1 uses MSE. v2-v5 use 0.9 MSE + 0.1 (1 - SSIM), with an 11 by 11 Gaussian SSIM window, sigma 1.5 and data range 1. AdamW uses learning rate 0.0003, weight decay 0.0001 and batch size 16. The planned comparison gives each run 15 epochs, with best validation checkpoint selection.')
    picture(ROOT/'outputs/comparison/improvement_validation_curves.png')
    p('Figure 2. Raw-input and standardized-input learning curves are separated because their reconstruction targets differ.','SmallBody')
    picture(run/'reconstructions_latest.png')
    p('Figure 3. Fixed model inputs, latest reconstructions and error. Scoring uses the best checkpoint. The reference reconstructs raw inputs; standardized ablations label their different input domain explicitly.','SmallBody')

    page();h('Phase 2 Scoring and statistical calibration')
    p('I fit 2,000 Isolation Forest trees, max_samples 256, on training latents only. Novelty is negative score_samples: higher means more unusual. Metadata and reconstruction errors are not forest features, and the library prediction offset does not determine the flags.')
    p('The original adjusted-boxplot threshold for v1 was 0.87484, above its maximum score 0.55648. Its multimodal histogram motivated a distribution-model revision; I retained the zero-flag result. I did not lower a threshold merely to produce five images.')
    p('For every run, I fit one through four Gaussian mixture components on calibration scores, select minimum BIC, and define T = max_j(mu_j + 3 sigma_j). All fitted populations contribute to the envelope. No assumed anomaly count or contamination fraction determines the result.')
    p(f"The selected run uses {calibration['components']} components, with fitted tail mass {calibration['fitted_tail_mass']:.6f}. This is a fitted-model description, not a real false-alarm guarantee. Correlated crops make BIC heuristic, and unknown contamination can distort the model.")
    if calibration['component_count_at_search_limit']:p('The component count reaches the search limit of four, which is a model-selection limitation.')
    picture(analysis/'score_distribution.png')
    p('Figure 4. Split score distributions, sorted novelty scores and the calibrated threshold with source-bootstrap uncertainty.','SmallBody')
    p(f"I resample whole calibration sources and refit component selection in 100 bootstrap repetitions. The validation CDF maximum distance is {diagnostics['validation_cdf_max_distance']:.4f}; I treat it descriptively and do not report an IID KS p-value. Conditional threshold flag frequency is not a probability of a genuine anomaly.")

    page();h('Did the improvement survive the checks?')
    p(escape(selection['robustness_summary']))
    stability=rows(ROOT/'outputs/comparison/run_stability.csv')
    comparisons={frozenset(pair) for pair in [('v3','robust_seed'),('v3','robust_split'),('v4','improved_seed'),('v4','improved_split')]}
    wanted=[r for r in stability if frozenset([r['left'],r['right']]) in comparisons]
    table([['Comparison','Rank correlation','Flag-set Jaccard']]+[[r['left']+' / '+r['right'],f"{float(r['spearman_all']):.3f}",f"{float(r['flag_jaccard']):.3f}" if r['flag_jaccard'] else 'Empty union'] for r in wanted],[270,110,110])
    p('Jaccard is intersection divided by union of flagged sets. It can be low even when global ranking correlation is high. Runs with different splits also report common-held-out comparisons in the CSV. Neither metric measures anomaly accuracy.')
    sensitivity=rows(ROOT/'outputs/comparison/photometric_sensitivity.csv')
    table([['Run','Median score change','95th percentile change','Flag flips / 512']]+[[r['run'],f"{float(r['median_absolute_score_change']):.6f}",f"{float(r['p95_absolute_score_change']):.6f}",r['flag_flips']] for r in sensitivity],[90,140,150,110])
    p('I hold each model, forest and threshold fixed, then apply 0.8*x + 0.1 outside its original inferred black border on a fixed validation sample. This tests a specific photometric nuisance while preserving the footprint, not correctness on labelled anomalies.')

    page();h('Why better brightness invariance is insufficient')
    p('All five leading v4 candidates have conspicuous black strips or borders. Their brightness distribution is less extreme, but acquisition support remains a competing explanation. These examples were inspected after statistical selection; they were not used to lower the boundary or select a desired count.')
    for name in ['v4','v5']:
        picture(ROOT/'outputs'/name/'mixture_three_sigma_trees2000/heatmap_sample_09852.png')
        p(name + ': the same validation crop after the independently calibrated selection. '+('The black border remains in the contrast-standardized input.' if name=='v4' else 'The footprint treatment leaves near-black edge fragments.')+' Error belongs to the standardized input domain.','SmallBody')
    p('The controlled perturbation favors the invariance designed into normalization. It establishes that mechanism, but cannot establish preservation of real albedo anomalies. I retain v3 as the screening reference and preserve both negative ablations for the engineering journal.')

    page();h('Phase 1.3 Latent-space diagnostics')
    p(f"The reference vector has covariance effective rank {diagnostics['effective_rank']:.2f} out of {diagnostics['latent_dimensions']}. It summarizes the variance spectrum, not a number of terrain classes.")
    for perplexity in [30,70]:
        picture(analysis/f'tsne_{perplexity}.png')
        p(f"Figure {5 if perplexity==30 else 6}. t-SNE at perplexity {perplexity}; novelty, supplied sun angle and five large source groups. PCA and standardization are for plotting only. The forest uses the full original latent.",'SmallBody')
    p(escape(selection.get('projection_interpretation','I use the two views to check whether broad structure persists, without treating branches as verified geological classes. Source and acquisition effects remain plausible.')))

    page();h('Phase 2.3 Metadata and candidate context')
    p('I join only through the supplied crop-to-source index. Latitude has 24 distinct values, longitude only 0 and 180. Resolution units and sun-angle semantics are unspecified. These fields do not justify precise localization, physical dimensions or causal claims.')
    for field in ['season','resolution']:
        data=rows(analysis/f'metadata_{field}.csv')
        table([[field,'Crops','Flags','Crop rate','Equal-source rate']]+[[r[field],r['count'],r['sum'],f"{float(r['mean']):.3%}",f"{float(r['equal_source_mean_flag_rate']):.3%}"] for r in data],[100,75,65,125,125])
    p(escape(selection['context_summary']))
    p('I keep metadata out of the novelty score and do not claim the optional fusion credit. Equal-source summaries reduce domination by large source observations, but they are descriptive.')

    for i,row in enumerate(chosen):
        page();h(f'Phase 3 Candidate {i+1}')
        item=hypotheses[row['filename']]
        p(f"<b>{escape(row['filename'])}</b> | {escape(row['source_image_id'])} | {escape(row['split'])}")
        p(f"Novelty {float(row['novelty_score']):.6f} exceeds {calibration['threshold']:.6f}. Conditional threshold-bootstrap flag frequency is {float(row['threshold_bootstrap_flag_frequency']):.1%}, not a posterior anomaly probability.")
        picture(analysis/f"heatmap_{Path(row['filename']).stem}.png")
        p(f'Figure {7+i}. Input, reconstruction, absolute error and discrepancy overlay. Standardized runs additionally show the raw crop. Error refers to the model-input domain and is not direct attribution for Isolation Forest.','SmallBody')
        p('<b>What I can see.</b> '+escape(item['evidence']))
        p('<b>Possible physical explanation.</b> '+escape(item['hypothesis']))
        p('<b>Alternative and limit.</b> '+escape(item['alternative']))
        p(f"Supplied metadata: latitude {row['latitude']}, longitude {row['longitude']}, sun angle {row['sun_angle']}, season {escape(row['season'])}, resolution {row['resolution']}.")
        if item.get('reference'):p('Process background, not crop identification: <link color="#1c5377" href="'+item['reference']+'">'+escape(item['reference'])+'</link>.','SmallBody')
    if not chosen:
        page();h('Phase 3 No crops pass the boundary')
        p('No image exceeds the calibrated boundary in this run. I retain that result and do not manufacture five examples. The notebook still records the score distribution and uncertainty.')

    page();h('Phase 4 My engineering notes')
    for paragraph in selection['student_journal'].split('\n\n'):
        paragraph=paragraph.strip()
        if paragraph.startswith('### '):p(escape(paragraph[4:]),'Heading2')
        elif paragraph:p(escape(paragraph.replace('**','')))
    page();h('References and reproducibility')
    refs=[('Competition inputs',"DA PS 26'.docx, DATA_DESCRIPTION.docx, RULES AND REGULATIONS.docx and the supplied archive."),
          ('SSIM','https://www.cns.nyu.edu/~lcv/ssim/'),('Resize-convolution','https://distill.pub/2016/deconv-checkerboard/'),
          ('Isolation Forest','https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html'),
          ('Gaussian mixtures','https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html'),
          ('Connected-component labelling','https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html')]
    for title,url in refs:
        p('<b>'+escape(title)+'.</b> '+('<link color="#1c5377" href="'+url+'">'+escape(url)+'</link>' if url.startswith('http') else escape(url)))
    p('The readable Colab notebook contains the preprocessing, model, training and scoring code directly. Completed configurations, checkpoints, manifests, per-epoch histories and numerical tables accompany the analysis. A fresh runtime needs the supplied private dataset ZIP. No pretrained weights or outside training images are used.')
    p('The recorded execution was local. Epoch wall times include interruptions and are not controlled hardware benchmarks. The independent checks are a limited sample of seeds and source assignments. Hidden labels would be needed to establish actual detection performance.')
    p('Private repository access, organizer collaborators and team details remain competition logistics. Local packaging does not submit the project.')
    target=ROOT/'output/pdf/Mars_HiRISE_Improved_Report.pdf';target.parent.mkdir(exist_ok=True,parents=True)
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#66717b'))
        canvas.drawString(45,26,'NSSC 2026 | Mars imagery | Preprocessing experiments')
        canvas.drawRightString(A4[0]-45,26,str(doc.page))
    doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=42,bottomMargin=43,title='Mars image anomaly detection: preprocessing experiments',author='')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(target)

if __name__=='__main__':main()
