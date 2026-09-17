"""Create the report from measured experiment outputs and reviewed hypotheses."""
from pathlib import Path
import csv,json
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]


def read_csv(path):
    with open(path,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    final=json.loads((ROOT/'outputs/final_selection.json').read_text())
    run=ROOT/'outputs'/final['run'];analysis=run/final['analysis']
    calibration=json.loads((analysis/'calibration.json').read_text())
    diagnostics=json.loads((analysis/'diagnostics.json').read_text())
    comparisons=read_csv(ROOT/'outputs/comparison/experiments.csv')
    hypotheses=json.loads((ROOT/'outputs/geological_hypotheses.json').read_text())
    selected=read_csv(analysis/'selected_for_interpretation.csv')
    target=ROOT/'output/pdf/Mars_HiRISE_Analysis_Report.pdf';target.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8,textColor=colors.HexColor('#20252b')))
    styles.add(ParagraphStyle(name='SmallText',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=6))
    styles['Title'].fontSize=24;styles['Title'].leading=29;styles['Title'].alignment=TA_LEFT
    styles['Heading1'].fontSize=16;styles['Heading1'].leading=20
    styles['Heading2'].fontSize=12;styles['Heading2'].leading=16
    story=[]
    def p(text,style='Body'):story.append(Paragraph(text,styles[style]))
    def heading(text):p(text,'Heading1')
    def picture(path,width=490):
        w,h=ImageReader(str(path)).getSize()
        story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,8))
    def table(rows,widths):
        data=[[Paragraph(escape(str(x)),styles['SmallText']) for x in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6edf3')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#cbd2d9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.append(t);story.append(Spacer(1,10))
    def newpage():story.append(PageBreak())

    p('Mars HiRISE anomaly detection','Title')
    p('NSSC 2026 | Measured experiments and statistical screening','Heading2')
    if final.get('status')!='Complete':p('<b>DRAFT: '+escape(final.get('status','Verification pending'))+'</b>')
    p('A from-scratch convolutional autoencoder compresses each grayscale crop into a fixed-length vector. Isolation Forest scores those vectors, and a calibrated distribution model supplies the anomaly boundary. The report documents the actual experiments, their limitations and the physical interpretation of the selected crops.')
    
    heading('Results and decision')
    p(f"<b>Selected configuration: {escape(final['run'])}.</b> {escape(final['rationale'])}")
    p(f"<font color='red'><b>Final novelty cutoff: {calibration['threshold']:.5f}.</b></font> {diagnostics['flagged_total']} of 10,422 crops exceed it. The 95% source-bootstrap threshold interval is {calibration['bootstrap_interval_95'][0]:.5f} to {calibration['bootstrap_interval_95'][1]:.5f}. Higher novelty means greater anomalousness. No assumed contamination count determines this result.")
    p('<b>Scope of the result.</b> Flags are candidates for investigation, not verified Genesis Outliers. There are no ground-truth labels, so precision, recall, F1 and AUROC cannot be computed. A heatmap explains reconstruction discrepancy, not the exact pixels responsible for the forest score.')
    p('<b>Major robustness limitation.</b> Independent initialization and source-split repeats have only 0.150 and 0.294 flag-set overlap with v3. No crop is flagged by all three pipelines. The five reviewed v3 examples are all unusually bright. The exact anomaly list is not robust.','SmallText')
    rows=[['Run','Latent','Val. MSE','Val. SSIM','Flags']]
    for row in comparisons:
        rows.append([row['run'],row['latent_dim'],f"{float(row['validation_mse']):.6f}",f"{float(row['validation_ssim']):.4f}",row.get('flagged','')])
    table(rows,[112,55,100,100,123])
    p('Metrics use each run\'s best validation-objective checkpoint. The split repeat uses different validation images, so its metrics are not a direct improvement comparison. Objectives differ across v1 and v2. Flag counts are not accuracy measures.','SmallText')
    p('The source documents and original archive remain unchanged. The local notebook, source code, configuration files, scores and changelog accompany this report. Repository access and team information remain submission logistics.','SmallText')

    newpage();heading('Data and validation protocol')
    p('Every supplied JPEG was decoded and checked. There are 10,422 grayscale images, all 227 by 227 pixels, with 10,422 unique crop-index entries and 172 unique source observations. All metadata joins resolve, and there are no exact decoded-pixel duplicates. Near duplicates and spatial overlap are not ruled out by that check.')
    table([['Split','Crops','Purpose'],['Training','7,101','Fit encoder and primary forest'],['Validation','1,666','Select reconstruction checkpoints; inspect density fit'],['Calibration','1,655','Fit score distribution and threshold']], [110,70,310])
    p('The main comparisons share source-disjoint partitions generated with split seed 2026. Roughly 70%, 15% and 15% of source IDs are allocated to these sets; crop proportions differ because source sizes range from 1 to 278. Additional robustness experiments, when listed, change the encoder seed or source split explicitly. The training set is unlabelled and must not be called clean-normal.')
    picture(ROOT/'outputs/audit/random_batch.png',width=315)
    p('Figure 1. Seeded 36-crop sanity sample. Appearance is not used to assign training anomaly labels.','SmallText')

    newpage();heading('Phase 1: Deep Latent Compression')
    p('Five 3 by 3 stride-two convolution blocks use channels 16, 32, 64, 96 and 128. Spatial sizes are 114, 57, 29, 15 and 8. Group normalization and SiLU follow each block. A flatten-and-linear layer creates a 128- or 256-dimensional vector. The decoder expands the vector to 128 by 8 by 8 and uses resize-convolution stages to recover 227 by 227, ending in a sigmoid.')
    p('All weights start randomly. There are no pretrained features, external pretraining images, or encoder-decoder skip connections. Intensities are divided by 255 with no per-image contrast normalization. The 128-dimensional version compresses 51,529 input values to 128 coordinates; 256 dimensions trades a larger code for more reconstruction capacity.')
    p('v1 uses MSE. v2 uses 0.9 MSE + 0.1 (1 - SSIM). SSIM uses an 11 by 11 Gaussian window, sigma 1.5 and data range 1, computed directly from image statistics. v3 retains the v2 loss and changes the bottleneck. All models use AdamW, learning rate 0.0003, weight decay 0.0001, batch size 16 and a maximum 15 epochs. Gradient clipping and mixed precision support local GPU training.')
    picture(ROOT/'outputs/comparison/validation_curves.png')
    p('Figure 2. Common validation metrics across the three controlled experiments. Lower MSE and gradient error are better; higher SSIM is better.','SmallText')
    picture(run/'reconstructions_latest.png')
    p('Figure 3. Fixed validation originals, latest-epoch reconstructions and absolute error. This panel shows the latest epoch, while scoring and selected-image explanations use the best checkpoint. Its shared panel error scale differs from the fixed [0,1] scale used in the final selected-image panels.','SmallText')

    newpage();heading('Phase 2: Isolation Forest Novelty Engine')
    p('The final Isolation Forest uses 2,000 trees, max_samples 256, all latent coordinates and contamination="auto". Increasing trees from 400 improved repeat-seed candidate agreement in the controlled v2 detector experiment. Only training latents fit the forest. The final score is negative score_samples. The library decision offset and predict method are not used to decide the flagged set.')
    p('The initial adjusted-boxplot rule failed as a useful screening boundary for v1: its cutoff was 0.87484, above the largest score of 0.55648. Its wide bootstrap interval and visibly multimodal score histogram motivated a distribution-model revision. The original zero-flag result is retained, rather than silently replaced.')
    p('The revised model fits one to four Gaussian mixture components to calibration scores, selects the lowest BIC, and defines T = max_j(mu_j + 3 sigma_j). The high-score component is not automatically declared anomalous. This envelope lies three standard deviations beyond every fitted population. No component weight or contamination fraction fixes the output count.')
    p(f"The selected run uses {calibration['components']} components. Its fitted tail mass beyond T is {calibration['fitted_tail_mass']:.6f}. Each Gaussian component contributes at most its weight times 0.00135 beyond this envelope. This describes the fitted model, not a guaranteed true false-alarm rate. Correlated crops also make BIC a heuristic rather than an independent-observation significance test.")
    if calibration['component_count_at_search_limit']:
        p('The selected component count reaches the search limit of four. For v3, the four-component BIC is only about 0.85 lower than the three-component BIC; model order is weakly separated. The group bootstrap repeats model selection and exposes some, but not all, of this uncertainty.','SmallText')
    picture(analysis/'score_distribution.png')
    p('Figure 4. Score distributions by split, sorted scores with source-bootstrap uncertainty, and empirical cumulative distribution.','SmallText')
    p(f"The held-out validation CDF maximum distance is {diagnostics['validation_cdf_max_distance']:.4f}. It is descriptive; no ordinary IID KS p-value is claimed. Every one of 100 bootstrap replicates resamples entire calibration sources and refits component selection. Encoder and forest uncertainty are assessed separately.")

    newpage();heading('Phase 2.5: Latent & Location Analysis')
    p(f"The selected latent has covariance effective rank {diagnostics['effective_rank']:.2f} out of {diagnostics['latent_dimensions']} coordinates. Effective rank summarizes the variance spectrum; it is not a count of geological classes or proof of collapse.")
    for perplexity in [30,70]:
        picture(analysis/f'tsne_{perplexity}.png')
        p(f'Figure {5 if perplexity==30 else 6}. t-SNE at perplexity {perplexity}, colored by novelty, supplied sun angle and five large source groups. Standardization and PCA to 50 dimensions are used only for plotting. The forest receives the original full latent.','SmallText')
    p('Both perplexities show a dense central region and narrow arms with higher novelty; this broad geometry persists while local placement changes. Source colors occupy the embedding unevenly, and acquisition or brightness effects may contribute. These visual structures are not confirmed geological classes. Interpret them with the numeric stability and candidate-context diagnostics.')

    stability=read_csv(ROOT/'outputs/comparison/run_stability.csv')
    table([['Runs compared','Rank correlation','Flag-set Jaccard']]+[[r['left']+' / '+r['right'],f"{float(r['spearman_all']):.3f}",f"{float(r['flag_jaccard']):.3f}" if r['flag_jaccard'] else 'No union'] for r in stability],[245,115,130])
    p('Jaccard is intersection divided by union of the flagged sets; rank correlation summarizes all scores. Neither is detection accuracy. A high rank correlation can coexist with unstable threshold membership. Runs with a changed source split also report correlations restricted to crops held out from both training sets in the accompanying CSV.')
    if final.get('robustness_summary'):p(escape(final['robustness_summary']))
    p('Metadata joins use only the supplied crop-to-source index. Latitude has 24 distinct values; longitude only 0 and 180. Sun-angle semantics and resolution units are not explicitly specified. Season is treated categorically. These coarse fields do not support precise geological localization or causal conclusions.')
    for field in ['season','resolution']:
        rows=read_csv(analysis/f'metadata_{field}.csv')
        table([[field,'Crops','Flagged','Crop rate','Equal-source rate']]+[[r[field],r['count'],r['sum'],f"{float(r['mean']):.3%}",f"{float(r['equal_source_mean_flag_rate']):.3%}"] for r in rows],[100,80,70,115,125])
    p('Equal-source rates give each observation the same weight. These descriptive comparisons expose unequal source sizes; they are not independent statistical tests. Optional metadata fusion is omitted so the required image-only pipeline remains the focus.')
    p('All five selected v3 crops exceed the 99.7th percentile of mean brightness, and three share SRC_154. Median crop-mean brightness is 217.90 among the 17 flags versus 122.63 among the other crops, on the original 0-255 scale. Whole-dataset score correlations cannot rule out a brightness effect specifically in the extreme tail. These examples may represent normal surface or acquisition variability; their appearance does not confirm an injected payload.')

    for i,row in enumerate(selected):
        newpage();heading(f"Phase 3: Candidate {i+1} Heatmaps & Hypotheses")
        item=hypotheses[row['filename']]
        p(f"<font color='blue'><b>Selected Anomaly: {escape(row['filename'])}</b></font> | {escape(row['source_image_id'])} | {escape(row['split'])} split")
        p(f"Novelty {float(row['novelty_score']):.5f} exceeds {calibration['threshold']:.5f}. Conditional threshold-bootstrap flag frequency: {float(row['threshold_bootstrap_flag_frequency']):.1%}. This frequency is not the probability of a genuine anomaly.")
        picture(analysis/f"heatmap_{Path(row['filename']).stem}.png")
        p(f'Figure {7+i}. Original, reconstruction, absolute-error map and overlay, with a fixed [0,1] error scale shared across all selected candidates.','SmallText')
        p('<b>Visible evidence.</b> '+escape(item['evidence']))
        p('<b>Physical hypothesis.</b> '+escape(item['hypothesis']))
        p('<b>Alternative explanation and limit.</b> '+escape(item['alternative']))
        if item.get('reference'):
            p('Process background, not an identification of this crop: <link href="'+item['reference']+'" color="#1c5377">'+escape(item['reference'])+'</link>.','SmallText')
        p(f"Provided metadata: latitude {row['latitude']}, longitude {row['longitude']}, sun angle {row['sun_angle']}, season {escape(row['season'])}, resolution {row['resolution']}. No catalog lookup or inferred units are added.")
    if not selected:
        newpage();heading('Phase 3: No thresholded candidates')
        p('No images exceed the final boundary. The threshold is not lowered to manufacture five examples.')

    newpage();heading('Phase 4: Architecture Iteration & Design Journal')
    for entry in final['journal_summary']:
        p(escape(entry['title']),'Heading2');p(escape(entry['text']))
    p('<b>Remaining limitations.</b> Hidden labels prevent ranking methods by actual anomaly recall. Fine-scale reconstruction remains imperfect. Gaussian mixtures can misfit tails or absorb contamination; limited source groups widen calibration uncertainty. Similar candidates may reflect ordinary terrain or acquisition footprints. The tested compute budget is finite and does not establish a global optimum.')
    p('A full symptom, diagnosis, fix and outcome journal is provided in ENGINEERING_CHANGELOG.md. Checkpoints, per-epoch histories and exact configurations are retained. Recorded wall times include local contention or host interruptions and are not controlled GPU benchmarks.')
    heading('References and provenance')
    references=[('Competition sources',"DA PS 26'.docx; DATA_DESCRIPTION.docx; RULES AND REGULATIONS.docx; supplied dataset archive."),('Wang et al. 2004, SSIM','https://www.cns.nyu.edu/~lcv/ssim/'),('Odena et al. 2016, resize-convolution','https://distill.pub/2016/deconv-checkerboard/'),('scikit-learn, Isolation Forest','https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html'),('scikit-learn, Gaussian mixture model selection','https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html'),('Hubert and Vandervieren 2008','https://doi.org/10.1016/j.csda.2007.11.008'),('Bouman and Heskes 2025','https://arxiv.org/abs/2501.13864')]
    for label,url in references:
        p(f'<b>{escape(label)}.</b> '+(f'<link href="{url}" color="#1c5377">{escape(url)}</link>' if url.startswith('https') else escape(url)),'SmallText')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#626b73'))
        canvas.drawString(45,26,'NSSC 2026 | Mars HiRISE anomaly detection')
        canvas.drawRightString(A4[0]-45,26,str(doc.page))
    document=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=42,bottomMargin=43,title='Mars HiRISE anomaly detection',author='')
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    print(target)

if __name__=='__main__':main()
