from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'
PDF=OUT/'DS-MINI-Design-울산_1반-김희윤.pdf'
_FONT_DIRS=[Path.home()/'Library/Fonts', Path('/Library/Fonts'), Path('/usr/share/fonts/truetype/nanum'), ROOT/'fonts', Path(__file__).resolve().parent/'fonts']
def _find_font(names):
    for d in _FONT_DIRS:
        for n in names:
            if (d/n).exists(): return str(d/n)
    raise FileNotFoundError('NanumGothic 폰트를 찾지 못했습니다: '+', '.join(names))
FONT=_find_font(['NanumGothic-Regular.ttf','NanumGothic.ttf']); BOLD=_find_font(['NanumGothic-Bold.ttf','NanumGothicBold.ttf'])
pdfmetrics.registerFont(TTFont('Nanum', FONT)); pdfmetrics.registerFont(TTFont('NanumBold', BOLD))

BLACK=colors.black; WHITE=colors.white; TABLE_NAVY=colors.HexColor('#17365D'); TABLE_PALE=colors.HexColor('#F4F8FB'); gray=colors.HexColor('#D9D9D9')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='KTitle',fontName='NanumBold',fontSize=23,leading=30,textColor=TABLE_NAVY,alignment=TA_CENTER,spaceAfter=7))
styles.add(ParagraphStyle(name='KSub',fontName='Nanum',fontSize=13,leading=18,textColor=colors.HexColor('#4F81BD'),alignment=TA_CENTER,spaceAfter=6))
styles.add(ParagraphStyle(name='KMeta',fontName='Nanum',fontSize=9,leading=13,textColor=colors.HexColor('#666666'),alignment=TA_CENTER,spaceAfter=16))
styles.add(ParagraphStyle(name='H1K',fontName='NanumBold',fontSize=16,leading=21,textColor=BLACK,spaceBefore=12,spaceAfter=6))
styles.add(ParagraphStyle(name='H2K',fontName='NanumBold',fontSize=12.5,leading=17,textColor=BLACK,spaceBefore=9,spaceAfter=4))
styles.add(ParagraphStyle(name='BodyK',fontName='Nanum',fontSize=9.5,leading=15,textColor=BLACK,spaceAfter=6))
styles.add(ParagraphStyle(name='SmallK',fontName='Nanum',fontSize=8.0,leading=11))
styles.add(ParagraphStyle(name='SmallHeaderK',fontName='NanumBold',fontSize=8.0,leading=11,textColor=WHITE))
styles.add(ParagraphStyle(name='CaptionK',fontName='Nanum',fontSize=8.5,leading=11,textColor=BLACK,alignment=TA_CENTER,spaceAfter=7))

def formalize(text):
    text=str(text)
    for a,b in [('세 가지다.','세 가지입니다.'),('후보 신호다.','후보 신호입니다.'),('탐색 지표다.','탐색 지표입니다.'),('-0.194였다.','-0.194였습니다.'),('약했다.','약했습니다.'),('이다.','입니다.'),('있다.','있습니다.'),('없다.','없습니다.'),('했다.','했습니다.'),('한다.','합니다.'),('된다.','됩니다.'),('필요하다.','필요합니다.'),('남긴다.','남깁니다.'),('선택한다.','선택합니다.'),('설계한다.','설계합니다.'),('보고한다.','보고합니다.')]:
        text=text.replace(a,b)
    return text
def glyph_safe(text):
    # NanumGothic에 Δ(U+0394)·−(U+2212) 글자가 없어 PDF에서 공백으로 보이는 문제 방지
    return str(text).replace('Δ','델타').replace('−','-')
def P(x,style='BodyK'):
    text=glyph_safe(formalize(x) if style=='BodyK' else str(x))
    return Paragraph(text.replace('\n','<br/>'),styles[style])
def table(headers, rows, widths, fs=7.8):
    data=[[P(h,'SmallHeaderK') for h in headers]]+[[P(v,'SmallK') for v in row] for row in rows]
    t=Table(data,colWidths=[w*mm for w in widths],repeatRows=1,hAlign='LEFT')
    st=[('BACKGROUND',(0,0),(-1,0),TABLE_NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),('FONTNAME',(0,0),(-1,0),'NanumBold'),('GRID',(0,0),(-1,-1),.35,gray),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]
    for i in range(1,len(data)):
        if i%2==0: st.append(('BACKGROUND',(0,i),(-1,i),TABLE_PALE))
    t.setStyle(TableStyle(st)); return t
def fig(name, caption, width=175*mm):
    from PIL import Image as PILImage
    source=PILImage.open(OUT/name)
    aspect=source.height/source.width
    # Preserve the original aspect ratio so the heatmap and its labels do not look compressed.
    height=width*aspect
    if height > 105*mm:
        height=105*mm
        width=height/aspect
    img=Image(str(OUT/name),width=width,height=height)
    return [img,P(caption,'CaptionK')]
def cover_page(canvas,doc):
    canvas.saveState(); canvas.setFillColor(WHITE); canvas.rect(0,0,A4[0],A4[1],fill=1,stroke=0); canvas.restoreState()
def footer(canvas,doc):
    canvas.saveState(); canvas.setFont('Nanum',8); canvas.setFillColor(BLACK); canvas.drawCentredString(A4[0]/2,12*mm,'ESS 배터리 수명 예측 | DAY 1 EDA·모델 설계 및 DAY 2 모델 평가'); canvas.restoreState()

story=[]
story += [Spacer(1,30*mm),P('ESS 배터리 수명 예측','KTitle'),P('DAY 1 EDA·회귀 모델 설계 및 DAY 2 모델 평가 보고서','KSub'),P('울산 1반 김희윤','KMeta'),P('MIT Stanford Battery Cycle Life Dataset | Batch 1 Batch 2 Batch 3','KMeta')]
story += [PageBreak(),P('1 분석 개요','H1K'),P('이 보고서는 ESS 기업의 운영·품질·교체 의사결정을 지원한다는 관점에서 Kaggle의 MIT Stanford 배터리 데이터셋을 분석한 결과입니다. 초기 배터리 상태와 충전 조건이 전체 Cycle Life와 어떤 관계를 가지는지 탐색하고, 조기 수명 예측 모델을 설계하는 것이 목적입니다.'),P('핵심 결론은 세 가지입니다. 첫째, Batch별 Cycle Life 분포가 크게 이동하므로 특정 Batch 내부 점수만으로 상용 모델을 선택하면 안 됩니다. 둘째, 초기 평균 QD와 Qdlin 기반 델타Q(V)가 수명 차이를 설명하는 중요한 후보 신호입니다. 셋째, C-rate와 온도는 열화 조건을 나타내지만 일부 Feature는 서로 중복되므로 안정적인 운영 모델을 위해 다중공선성 점검이 필요합니다.')]
story += [table(['항목','정의','이번 분석에서의 처리'],[['분석 대상','Batch 1 + Batch 2 + Batch 3','파일별 셀 단위와 Cycle 단위 분리'],['관측 단위','배터리 셀','초기 100 Cycle을 셀 1행으로 집계'],['Target','cycle_life','EOL 80% 용량 도달까지 전체 Cycle 수'],['문제 유형','Regression','연속형 Cycle Life 직접 예측'],['공식 평가 지표','MAPE','Train CV, Valid Hold-out, Batch 2 Test 비교']],[30,48,105]),Spacer(1,4)]
story += [P('2 데이터 구조와 전처리','H1K'),P('원본은 MATLAB v7.3 구조로 저장되어 있으며 하나의 셀 안에 cycle_life, 충전정책, summary, cycles가 함께 들어 있다. summary는 Cycle별 QD, IR, 온도, 충전시간을 제공하고, cycles의 Qdlin은 전압축으로 선형 보간된 방전용량 곡선이다.')]
life=pd.read_csv(OUT/'day1_final_q1_summary.csv',index_col=0)
life_counts=pd.read_csv(OUT/'day1_final_life_all_batches.csv')
batch_totals=life_counts.groupby('batch').size().to_dict()
batch_valid=life_counts[life_counts['target_valid']==True].groupby('batch').size().to_dict()
batch_missing={b:int(batch_totals.get(b,0)-batch_valid.get(b,0)) for b in batch_totals}
mean_life={b:float(r['mean']) for b,r in life.iterrows()}
story += [table(['Batch','원본 파일','전체 셀','유효 Target','결측·미확정','평균 Cycle Life','역할'],[['Batch 1','2017-05-12',str(batch_totals.get('Batch 1',0)),str(batch_valid.get('Batch 1',0)),str(batch_missing.get('Batch 1',0)),f"{mean_life.get('Batch 1',float('nan')):.1f}",'학습 데이터'],['Batch 2','2018-02-20',str(batch_totals.get('Batch 2',0)),str(batch_valid.get('Batch 2',0)),str(batch_missing.get('Batch 2',0)),f"{mean_life.get('Batch 2',float('nan')):.1f}",'최종 테스트'],['Batch 3','2018-04-12',str(batch_totals.get('Batch 3',0)),str(batch_valid.get('Batch 3',0)),str(batch_missing.get('Batch 3',0)),f"{mean_life.get('Batch 3',float('nan')):.1f}",'추가 검증']],[20,30,18,23,24,30,35]),P('<b>회사 관점의 표 인사이트</b> Batch 1은 전체 46셀 중 10셀이 0.88 Ah EOL에 도달하기 전에 기록이 종료되어 유효 Target 36셀만 학습에 사용했습니다. Batch 2는 유효 39셀, Batch 3은 유효 44셀입니다. 수명 미확정 셀을 억지로 수명값으로 대체하지 않았기 때문에 모델 평가의 정답 신뢰성을 지킬 수 있습니다. 회사는 향후 실험·운영 데이터에서 EOL 도달 여부와 기록 종료 사유를 별도 품질 항목으로 관리해야 합니다.')]
story += [P('cycle_life가 없거나 0.88 Ah EOL에 도달하기 전에 기록이 끝난 셀은 실제 수명을 알 수 없으므로 수명 통계와 회귀 평가에서 제외했습니다. Summary의 첫 행 QD=0은 실제 용량 0이 아니라 측정 누락 Sentinel 값으로 보고 QD>0인 관측만 Feature 계산에 사용했습니다. 또한 IR이 0으로 기록된 셀은 IR 결측 Sentinel로 처리하고, 해당 셀 자체는 제외하지 않되 IR을 보조 Feature로만 사용했습니다. 짧은 수명 셀은 임의 삭제하지 않고 Batch와 충전정책 차이를 확인할 대상으로 남겼습니다.')]
story += [P('3 EDA 질문별 분석 결과','H1K'),P('각 질문을 분석 목적, 실제 확인 결과, 해석, 모델링 시사점 순서로 정리했다. 그래프는 단순 나열하지 않고 다음 단계의 변수 설계와 검증 방식으로 연결했다.')]

life=pd.read_csv(OUT/'day1_final_q1_summary.csv',index_col=0)
life_counts=pd.read_csv(OUT/'day1_final_life_all_batches.csv')
batch_totals=life_counts.groupby('batch').size().to_dict()
batch_valid=life_counts[life_counts['target_valid']==True].groupby('batch').size().to_dict()
batch_missing={b:int(batch_totals.get(b,0)-batch_valid.get(b,0)) for b in batch_totals}
mean_life={b:float(r['mean']) for b,r in life.iterrows()}
story += [P('3.1 Cycle Life 분포는 어떻게 생겼는가','H2K'),P('과제 기준인 500 Cycle 미만과 1,000 Cycle 초과를 기본 분포 기준으로 사용하고, 참고 보고서와 비교할 수 있도록 550 Cycle 미만 결과도 민감도 기준으로 함께 표시했다. 분석 범위는 150~2,300 Cycle이다.')]+fig('day1_final_q1.png','그림 1. 150~2,300 Cycle 범위의 Batch별 Cycle Life 분포와 Box Plot')
rows=[]
for b,r in life.iterrows(): rows.append([b,int(r['count']),f"{r['mean']:.1f}",f"{r['median']:.1f}",f"{r['min']:.0f}~{r['max']:.0f}",f"{int(r['long_gt_1000'])} ({r['long_ratio']*100:.1f}%)",f"{int(r['short_lt_500'])} ({r['short_ratio']*100:.1f}%) / {int(r['short_lt_550'])} ({r['short_550_ratio']*100:.1f}%)"])
story += [table(['Batch','유효 셀','평균','중앙값','범위','장수명 >1,000','단수명 <500 / <550'],rows,[20,20,22,23,30,34,43]),P(f'<b>표·그래프 인사이트 및 회사 시사점</b> 유효 Target 기준으로 Batch 1은 {mean_life.get("Batch 1",0):.1f} Cycle, Batch 2는 {mean_life.get("Batch 2",0):.1f} Cycle, Batch 3은 {mean_life.get("Batch 3",0):.1f} Cycle의 평균 수명을 보입니다. 과제 기준인 500 Cycle 미만 셀은 Batch 1 0개, Batch 2 28개, Batch 3 0개이며, 참고 기준인 550 Cycle 미만에서는 각각 1개, 30개, 1개입니다. 따라서 Batch 2에 짧은 수명 셀이 집중되어 있다는 결론은 두 기준에서 동일합니다. 회사는 단일 수명 기준만으로 셀을 평가하지 말고 Batch와 충전 조건을 함께 반영해 출하 전 위험군과 원인 분석 대상을 구분해야 합니다. 수명 미확정 셀은 이상치로 삭제한 것이 아니라 정답이 없는 셀로 분리했습니다. IQR 기준 이상치는 Batch 1에 없고, Batch 2의 9개와 Batch 3의 3개는 모두 newstructure 셀로 유독 긴 쪽입니다. 가장 짧은 셀은 Batch 2의 392~416 Cycle 셀들로, Batch 2에서는 짧은 수명이 일반적인 경우입니다. Batch 1에서 가장 짧은 4개 셀은 80%까지 평균 충전속도가 5.1C로 나머지(4.5C)보다 빨랐고, 온도와 내부저항은 차이가 없어 고속 충전과 더 관련이 있습니다.'),P('<b>모델링 시사점</b> Batch 1 내부 무작위 분할만으로 모델을 선택하지 않고 Batch 2를 독립 Test로 둡니다. Batch ID를 단순 Feature로 넣기보다 분할 설계에서 새로운 Batch에 대한 일반화 성능을 평가합니다.')]

knee=pd.read_csv(OUT/'day1_final_q2_knee_summary.csv')
story += [PageBreak(),P('3.2 배치별 Cycle에 따른 방전용량은 어떻게 감소하는가','H2K'),P('각 셀의 x축은 사용 Cycle, y축은 방전용량 QD(Ah)입니다. 색은 Cycle Life를 나타내며, 빨간 점선은 예측에 사용하는 100번째 Cycle, 회색 점선은 EOL 기준 0.88Ah입니다. Knee는 곡선을 15 Cycle 중앙값으로 평활한 뒤 시작점과 끝점을 잇는 직선에서 가장 멀리 떨어지는 지점으로 정의했습니다.')]+fig('day1_final_q2.png','그림 2. 배터리별 Cycle에 따른 방전용량: 색 = Cycle Life')
story += [table(['Batch','유효 셀','Knee 중앙값','수명 대비 Knee','100번째 전 Knee'],[[r['batch'],int(r['valid_cells']),f"{r['knee_cycle_median']:.0f}",f"{r['knee_life_ratio_median']*100:.0f}%",int(r['knee_before_100_count'])] for _,r in knee.iterrows()],[25,25,35,40,40]),P('<b>표·그래프 인사이트 및 회사 시사점</b> 모든 Batch에서 100번째 Cycle 이전에 Knee가 나타난 셀은 없습니다. 따라서 처음 100 Cycle에서는 용량 곡선이 급격히 꺾이는 시점 자체를 직접 관측할 수 없습니다. 회사는 이후의 Knee 위치를 예측변수로 사용할 수 없으므로, 초기 용량 수준과 ΔQ(V)처럼 100 Cycle 안에서 관측 가능한 변화량을 사용해야 합니다. Batch별 Knee 중앙값은 약 '+', '.join(f"{v:.0f}" for v in knee['knee_cycle_median'])+' Cycle로 다르며, 배터리마다 열화 경로가 달라 단일 교체 기준을 적용하기 어렵습니다. 열화 속도는 일정하지 않고 가속됩니다. Cycle 10~100의 용량 기울기는 단수명(550 미만)과 장수명(1,000 초과) 모두 0에 가깝지만, EOL 직전 100 Cycle에서는 단수명 -1.55, 장수명 -0.75 mAh/cycle로 빨라집니다.'),P('<b>모델링 시사점</b> Knee 위치는 사후 설명용 지표로만 사용하고, 초기 100 Cycle의 QD 요약값과 ΔQ(V) Feature를 모델에 사용합니다.')]

q3=pd.read_csv(OUT/'day1_final_q3_summary.csv'); q3rows=[[r['batch'],int(r['n']),f"{r['corr_delta_var_cycle_life']:+.2f}",f"{r['corr_log_delta_var_log_cycle_life']:+.2f}"] for _,r in q3.iterrows()]
story += [PageBreak(),P('3.3 초기 델타Q(V) 변화에서 차이가 나타나는가','H2K'),P('원본 cycles[n][\'Qdlin\']의 1,000개 전압 포인트(2V~3.6V)를 사용해 델타Q(V) = Qdlin(Cycle 100,V) − Qdlin(Cycle 10,V)를 계산합니다. 곡선의 분산과 수명을 원래 값과 로그 변환 값으로 각각 비교했습니다. 그래프의 아래 패널은 세 Batch 전체 셀을 합쳐 log10(델타Q 분산)과 log10(Cycle Life)의 관계를 표시하며, 점선은 전체 직선 적합입니다.')]+fig('day1_final_q3.png','그림 3. 배치별 델타Q(V) 곡선과 log 분산·log 수명 관계')
story += [table(['Batch','셀 수','원래 값 상관','로그 변환 상관'],q3rows,[35,25,45,50]),P('<b>표·그래프 인사이트 및 회사 시사점</b> 로그 변환 후 전체 상관계수는 약 -0.90으로, 델타Q(V) 분산이 클수록 수명이 짧아지는 관계가 가장 뚜렷해집니다. 이 관계는 Batch 1 -0.84, Batch 2 -0.92, Batch 3 -0.76으로 Batch가 바뀌어도 유지됩니다. 회사는 초기 100 Cycle에서 Q(V) 곡선 변화가 크고 불안정한 셀을 조기 위험군으로 분류할 수 있습니다. 다만 Batch 3의 Qdlin 시작점 차이는 배치 효과일 수 있으므로 원인으로 단정하지 않습니다.'),P('<b>모델링 시사점</b> ΔQ(V) 전체 곡선을 그대로 넣지 않고 분산(log), 최솟값 또는 평균 중 하나를 핵심 Feature로 선택하고, 수명에도 log를 적용한 회귀를 우선 검토합니다.')]

q4=pd.read_csv(OUT/'day1_final_q4_summary.csv')
protocol=pd.read_csv(OUT/'day1_final_q4_protocol_summary.csv')
current_corr=pd.read_csv(OUT/'day1_final_q4_current_corr.csv',index_col=0)
story += [PageBreak(),P('3.4 충전 조건과 수명의 관계','H2K'),P('충전정책에서 첫 번째 속도 C1, 전환 지점 Q1, 두 번째 속도 C2를 추출합니다. 80%까지 평균 충전속도는 단순한 C-rate 평균이 아니라 충전시간의 역수로 계산합니다. Cavg_80 = 0.8 ÷ (Q1/C1 + (0.8-Q1)/C2)입니다. 예를 들어 8C(15%)-3.6C는 약 4.0C이고, 5.4C(80%)-5.4C는 5.4C입니다.')]+fig('day1_final_q4.png','그림 4. Batch 1 충전 방식별 Mean Cycle Life(x)와 Avg charging speed to 80%(y), 그리고 배치별 평균속도·수명 관계')
story += [P('<b>그래프 인사이트 및 회사 시사점</b> 왼쪽 산점도는 Batch 1의 유효 36셀을 충전 방식별로 평균 내어 Mean Cycle Life를 x축, Cavg_80을 y축으로 표시합니다. 1단계 C-rate만 빠른 방식이라도 80%까지의 대부분을 느린 속도로 충전하면 Cavg_80은 낮아질 수 있습니다. 오른쪽 산점도는 셀 단위 Cavg_80과 Cycle Life를 Batch별로 비교하며, Batch 1 내부에서는 빠른 평균 충전속도와 짧은 수명이 함께 나타나는 경향이 있지만 Batch를 합치면 배치 효과가 섞입니다. 따라서 회사는 충전속도를 원인으로 단정하기보다 운영 조건별 위험 신호로 관리해야 합니다. 충전 전류 패턴과 초기 100 Cycle 열화 속도의 상관은 2단계 C-rate 0.25, 평균 충전 전류 0.25, C-rate 변화량 0.20으로 약한 양의 상관이고, 1단계 C-rate는 -0.12입니다. 처음에 세게 충전하는지보다 충전 전체의 평균 전류가 클수록 초기 열화가 조금 빠르지만, 관계가 약해 열화 속도만으로 수명을 설명하기는 어렵습니다.'),P('<b>모델링 시사점</b> charging_policy 문자열 자체 대신 C1, Q1, C2, Cavg_80과 초기 충전시간을 구조화 Feature로 만들고, Batch 1에서만 보이는 관계인지 독립 Batch에서 재확인합니다.')]

corr=pd.read_csv(OUT/'day1_final_q5_log_corr.csv',index_col=0)
story += [P('3.5 어떤 초기 신호가 수명과 연관되는가','H2K'),P('초기 100 Cycle에서 계산한 Feature와 log10(Cycle Life)의 Pearson 상관을 Batch 1과 전체 Batch에 대해 비교했습니다. Feature를 고르는 기준은 학습 Batch에서 강하고, 다른 Batch에서도 방향이 유지되며, 서로 중복되지 않는지입니다.')]+fig('day1_final_q5.png','그림 5. log 수명과 초기 100 Cycle Feature 상관관계')
story += [table(['Feature','Batch 1 상관','전체 상관','판단'],[[idx,f"{r['Batch 1']:+.2f}",f"{r['전체']:+.2f}",'핵심' if idx=='log_delta_var' else ('중복(핵심과 같은 ΔQ 정보)' if idx.startswith('delta_') else ('Batch 1 보조' if abs(r['Batch 1'])>=.4 and abs(r['전체'])<.3 else '약함'))] for idx,r in corr.iterrows()],[58,32,30,35]),table(['다중공선성 후보','처리 방향'],[['ΔQ 분산·log 분산·최솟값·평균','같은 곡선 정보이므로 핵심 1개만 선택'],['초기 충전시간·Cavg_80','서로 중복되므로 하나만 선택'],['평균 온도·최고 온도','둘 중 하나 선택 또는 Ridge 정규화'],['QD 감소량·감소기울기','둘 중 하나 선택']],[70,85]),P('<b>표·그래프 인사이트 및 회사 시사점</b> log ΔQ 분산은 Batch 1에서 -0.84, 전체에서 -0.90으로 가장 안정적인 핵심 단서입니다. 반면 Cavg_80은 Batch 1에서만 -0.59로 관련되고 전체에서는 -0.15로 약해져, 충전속도만으로 배치가 다른 배터리의 수명을 판단하면 위험합니다. 회사는 ΔQ 기반 위험 신호를 주력으로 사용하고, 충전속도·온도·초기 용량은 보조 정보로 검증해야 합니다.')]

story += [P('4 Feature Engineering 전략','H1K'),P('Feature는 상관계수가 큰 변수를 모으는 방식이 아니라 배터리 열화의 물리적 의미와 EDA 결과를 함께 반영해 설계한다. 모든 Feature는 초기 100 Cycle에서 계산해 미래 정보가 Target에 섞이는 것을 방지한다.'),table(['Feature 그룹','원본 변수','생성 Feature','선정 근거','주의점'],[['용량 열화','QDischarge','QD_mean, QD_std, QD_slope, QD_first, QD_last','초기 수준과 감소 속도','QD 파생 Feature 중복'],['전압별 열화','Qdlin','delta_qv_mean, abs_mean, max_abs, 구간 평균','단수명군 델타Q(V) 증가','고차원 곡선 축약'],['전기적 상태','IR','IR_mean, IR_std, IR_change','전기적 상태','초기 IR 단독 상관 약함'],['열 스트레스','Tavg, Tmax, Tmin','Tavg_mean, Tmax_mean, T_range','열화 조건','Tavg·Tmax 중복'],['충전 조건','policy, charge time','first C-rate, second C-rate, SOC 전환, policy family','C-rate별 수명 차이','정책별 표본 수']],[25,32,60,42,38])]

story += [P('5 Regression 선택과 모델링 전략','H1K'),P('Target인 cycle_life는 EOL까지의 전체 Cycle 수라는 연속값이므로 Regression을 선택합니다. Classification은 장수명·단수명 선별에는 유용하지만 교체 시점에 필요한 수명 규모 정보를 잃습니다.'),P('<b>기업 관점에서 Regression을 선택한 이유</b> ESS 기업은 배터리를 단순히 정상·불량으로 나누는 것보다 각 배터리가 앞으로 몇 Cycle을 사용할 수 있는지 알아야 합니다. 수명 숫자를 예측하면 배터리별 잔여 운영기간을 추정하고, 교체 부품과 정비 인력을 미리 계획하며, 수명이 짧은 셀을 조기에 선별할 수 있습니다. 또한 충전정책별 수명 손실을 비교해 빠른 충전의 운영 편의성과 교체비용 사이의 균형을 판단할 수 있습니다. 따라서 Regression은 기업의 유지보수 일정, 재고·비용 계획, 운영 위험 관리를 직접 지원한다는 이점이 있어 선택합니다.'),table(['후보 모델','선정 근거','확인할 한계','사용 목적'],[['DummyRegressor','평균 수명 기준선','설명력 없음','최소 비교 기준'],['Ridge Regression','다중공선성 완화와 계수 해석','비선형 패턴 한계','해석 가능한 기준 모델'],['Random Forest','비선형 관계와 상호작용','비선형 관계 과대적합 가능성','비선형 구조 확인'],['HistGradientBoosting 또는 LightGBM','표 형태 Feature의 비선형 조합','Batch 2 일반화 확인','최종 후보 비교']],[42,62,50,38]),table(['EDA 확인 내용','전처리·Feature 전략','모델링 반영'],[['Batch별 Target 분포 이동','Batch ID를 일반 Feature로 넣지 않음','Batch 1 학습, Batch 2 Test'],['QD 열화 속도와 Knee 차이','초기 QD 수준·표준편차·기울기','선형·비선형 비교'],['단수명군 델타Q(V) 증가','Qdlin 곡선을 통계량으로 축약','추가 Feature 세트 비교'],['C-rate와 평균 수명 감소','C-rate·SOC 전환 구조화','정책별 성능 확인'],['Tavg와 Tmax 높은 상관','Feature 제거 또는 Ridge','계수 안정성과 Tree 중요도 비교']],[58,65,68])]

split=pd.read_csv(OUT/'day2_split_summary.csv').iloc[0]
story += [P('6 데이터 분할과 성능 평가 계획','H1K'),P('배터리 데이터는 셀 단위로 독립적이며, 셀마다 다른 충전 프로토콜로 실험되었습니다. 같은 프로토콜의 셀이 Train과 Valid에 나뉘어 들어가면 프로토콜 정보를 공유하는 누수가 생기므로, 과제 기준에 따라 Valid는 CV가 아닌 Hold-out으로 두고 <b>충전 프로토콜 단위</b>로 분리합니다. Train CV도 같은 이유로 프로토콜 단위 GroupKFold를 사용합니다.'),table(['구분','데이터','목적','보고 지표'],[['Train',f"Batch 1 {int(split['cv_folds'])}-fold GroupKFold CV",'모델 후보 학습과 평균 성능','평균 MAPE'],['Valid','Batch 1 프로토콜 단위 Hold-out','과적합과 Feature 점검','MAPE'],['Test','Batch 2 유효 Target 셀','다른 Batch 일반화','MAPE'],['추가 Test','Batch 3 유효 Target 셀','분포가 다른 Batch 검증','MAPE'],['Gap','Train-Valid, Valid-Test, Batch 2-Batch 3','과적합·Batch 이동 해석','차이값']],[25,48,68,35]),P('공식 지표는 MAPE로 통일합니다. ESS 운영에서는 수명을 실제보다 길게 예측하는 것이 위험할 수 있으므로 MAPE와 함께 과대예측 비율과 평균 signed error를 보조 지표로 확인합니다. 원논문의 Regression 기준 9.1%와 Batch 2 Test 성능의 Gap도 별도로 보고합니다.')]
story += [P(f"코드 구현 기준으로 Batch 1 유효 셀 36개를 Train {int(split['n_train'])}셀({int(split['n_train_protocols'])}개 프로토콜)과 Valid {int(split['n_valid'])}셀({int(split['n_valid_protocols'])}개 프로토콜)로 나눴고, 두 집합 사이에 겹치는 프로토콜은 없습니다. CV와 모델 학습은 Train에만 수행하고 Valid는 모델 선택 확인에만 사용합니다. Batch 2 39셀은 최종 Test, Batch 3 44셀은 추가 Test로, 모델 선택이 끝난 뒤 한 번만 평가합니다.")]

model=pd.read_csv(OUT/'final_model_comparison.csv')
errs=pd.read_csv(OUT/'day2_error_by_life_range.csv')
prange=pd.read_csv(OUT/'day2_prediction_range.csv')
def short(name):
    return (name.replace('HistGradientBoosting','HistGB').replace('Dummy 평균 기준선','Dummy 기준선')
                .replace(' (DAY1 전략 세트(10))',' · 전략 10').replace(' (전체 세트(18))',' · 전체 18'))
model_rows=[]
for _,r in model.iterrows():
    decision='최종 선택' if bool(r['selected_for_business']) else ('기준선' if 'Dummy' in r['model'] else '비교 후보')
    model_rows.append([short(r['model']),f"{r['rep_valid_mean']*100:.1f}%",f"{r['cv_mape']*100:.1f}%",f"{r['valid_mape']*100:.1f}%",f"{r['batch2_test_mape']*100:.1f}%",f"{r['batch3_test_mape']*100:.1f}%",decision])
best=model.iloc[0]
cand=model[~model['model'].str.contains('Dummy')]
tied=cand[cand['rep_valid_mean']<=cand['rep_valid_mean'].min()+0.005]
test_best=cand.sort_values('batch2_test_mape').iloc[0]
stab=pd.read_csv(OUT/'day2_split_stability.csv'); imp=pd.read_csv(OUT/'day2_feature_importance.csv')
single_top=stab['single_split_choice'].value_counts().iloc[0]
def er(batch,seg,col):
    q=errs[(errs['batch']==batch)&(errs['구간']==seg)]
    return float(q[col].iloc[0]) if len(q) else float('nan')
b2_in, b2_lo, n_b2_lo = er('Batch 2','학습 범위 안','MAPE'), er('Batch 2','학습 최솟값 미만','MAPE'), int(er('Batch 2','학습 최솟값 미만','셀수'))
b3_in, b3_hi, n_b3_hi = er('Batch 3','학습 범위 안','MAPE'), er('Batch 3','학습 최댓값 초과','MAPE'), int(er('Batch 3','학습 최댓값 초과','셀수'))
tr_lo, tr_hi = prange.iloc[0]['min'], prange.iloc[0]['max']
pr_lo, pr_hi = prange.iloc[3]['min'], prange.iloc[3]['max']
sel_txt=(f"Valid가 {int(split['n_valid'])}셀뿐이라 분할 하나만 보고 고르면 분할마다 선택이 바뀝니다(분할 20개에 적용했을 때 가장 많이 뽑힌 모델도 {single_top}/20회). "
         f"그래서 Batch 1 안에서 서로 다른 프로토콜 단위 Hold-out 분할 {len(stab)}개를 만들어 평균 Valid MAPE가 가장 낮은 모델을 고르고, 0.5%p 이내 동률이면 평균 CV로 결정했습니다. "
         f"그 결과 {short(best['model'])} 모델이 반복 평균 Valid {best['rep_valid_mean']*100:.2f}% ± {best['rep_valid_std']*100:.2f}%로 선택되었습니다. "
         "DAY1에서 설계한 전략 세트가 최종 선택되어, 중복 Feature를 줄이는 전략이 실제 성능으로 뒷받침되었습니다.")
leak_txt=''
story += [PageBreak(),P('7 모델 개발 및 성능 평가','H1K'),
          P(f"앞선 EDA 결과를 실제 Pipeline에 반영했습니다. 학습 데이터는 유효 Target이 확인된 Batch 1의 36셀이고, 충전 프로토콜 단위로 Train {int(split['n_train'])}셀과 Hold-out Valid {int(split['n_valid'])}셀을 나눴습니다. Batch 2의 39셀은 독립 Test, Batch 3의 44셀은 추가 검증입니다. 결측 대체·표준화 통계량은 Train에만 적합했습니다. DAY1 전략대로 중복 Feature를 하나씩만 남긴 <b>전략 세트(10개)</b>와 비교용 <b>전체 세트(18개)</b>를 모든 후보 모델에 적용해 비교했습니다."),
          table(['모델 · Feature 세트','반복 Valid 평균','고정 CV','고정 Valid','Batch 2 Test','Batch 3 Test','판단'],model_rows,[42,20,17,18,20,20,19]),
          P('반복 Valid 평균은 Batch 1 Hold-out 분할 20개 평균, 고정 CV·Valid는 지정 성능 포맷에 쓴 분할 1개 기준입니다.','CaptionK'),
          P(f"<b>모델 선택 결론</b> {sel_txt} Batch 2·3 결과는 선택이 끝난 뒤 확인한 일반화 성능입니다.{leak_txt} HistGB는 학습 셀이 30개 안팎이므로 min_samples_leaf=3으로 설정해 실제로 트리가 분기하도록 했습니다."),
          P(f"<b>성능 인사이트 및 회사 시사점</b> Batch 2 Test MAPE는 {best['batch2_test_mape']*100:.1f}%로 Valid {best['valid_mape']*100:.1f}%보다 크게 높습니다. 원인은 예측 범위에서 확인됩니다. 최종 모델의 Batch 2·3 예측값은 {pr_lo:.0f}~{pr_hi:.0f} Cycle에 머무는데, 학습 셀의 수명이 {tr_lo:.0f}~{tr_hi:.0f} Cycle이라 이 범위 밖은 구조적으로 맞히지 못합니다. "
            f"실제로 Batch 2에서 학습 최솟값보다 짧은 {n_b2_lo}셀의 MAPE는 {b2_lo:.1f}%(전부 과대예측)지만, 학습 범위 안의 셀은 {b2_in:.1f}%입니다. Batch 3도 학습 범위 안 {b3_in:.1f}%, 학습 최댓값보다 긴 {n_b3_hi}셀 {b3_hi:.1f}%(과소예측)입니다. "
            f"즉 학습 범위 안에서는 원논문 9.1%와 비슷한 수준이고, 성능 저하는 Batch 간 수명 분포 이동에서 옵니다. Batch 2 과대예측 비율이 {best['batch2_overprediction_rate']*100:.1f}%이므로 그대로 쓰면 교체가 늦어질 위험이 있어, 보수적 안전계수와 위험군 재검사를 포함한 의사결정 보조 모델로 운영해야 합니다."),
          P('<b>전략 → 구현 점검</b> log ΔQ 분산, 초기 QD 수준·변동·감소량, IR, 온도, Cavg_80과 C-rate 구조 Feature가 코드에 반영되었고, 중복 Feature 처리 전략은 전략 세트와 전체 세트 비교로 검증했습니다. Target은 log10(cycle_life)로 학습하고 성능은 원래 Cycle 단위로 역변환해 보고했습니다. 모델 선택에는 Batch 1 Train의 CV와 Hold-out Valid만 사용했습니다.')]
sd=stab[['final_valid_mape','final_batch2_mape','final_batch3_mape']].agg(['mean','std','min','max']).T
stab_rows=[[n,f"{r['mean']:.2f}%",f"{r['std']:.2f}%p",f"{r['min']:.1f}~{r['max']:.1f}%"] for n,(_,r) in zip(['Valid (Batch 1 Hold-out)','Test (Batch 2)','Test (Batch 3)'],sd.iterrows())]
imp_rows=[[r['feature'],f"{r['mape_increase_pctp']:.2f}%p",f"{r['std']:.2f}"] for _,r in imp.head(6).iterrows()]
dq_sum=imp[imp['feature'].isin(['log_delta_var','delta_max_abs','delta_abs_mean'])]['mape_increase_pctp'].sum()
rest_sum=imp[~imp['feature'].isin(['log_delta_var','delta_max_abs','delta_abs_mean'])]['mape_increase_pctp'].clip(lower=0).sum()
story += [P('<b>분할 안정성</b> 최종 모델 구성을 서로 다른 Hold-out 분할 20개에서 다시 학습했습니다. 고정 분할의 결과가 아래 범위 안에 있어 특정 분할에서 우연히 나온 값이 아닙니다. Batch 3는 분할이 바뀌어도 안정적이고, Batch 2는 단수명 셀 예측이 학습 셀 구성에 민감해 더 흔들립니다.'),
          table(['구분 (분할 20개)','평균 MAPE','표준편차','범위'],stab_rows,[55,30,30,40]),
          P(f"<b>핵심 Feature 확인</b> Batch 1 안에서 프로토콜 단위 GroupKFold로 학습에 쓰지 않은 fold의 Feature 값을 섞어 MAPE 증가량을 계산했습니다(Permutation Importance). log_delta_var가 1위이며, 델타Q 계열 Feature의 기여({dq_sum:.2f}%p)가 나머지 Feature 전체({rest_sum:.2f}%p)보다 {dq_sum/max(rest_sum,1e-9):.1f}배 큽니다. DAY1 EDA에서 고른 핵심 신호가 모델에서도 그대로 확인됩니다."),
          table(['Feature','MAPE 증가','표준편차'],imp_rows,[60,35,30])]

reporting=pd.read_csv(OUT/'day2_regression_reporting.csv')
reporting_rows=[[r['구분'],f"{r['MAPE(%)']:.2f}%",r['비고']] for _,r in reporting.iterrows()]
batch3_reporting=pd.read_csv(OUT/'day2_batch3_reporting.csv')
batch3_rows=[[r['구분'],f"{r['MAPE(%)']:.2f}%",r['비고']] for _,r in batch3_reporting.iloc[len(reporting):].iterrows()]
story += fig('day2_performance.png','그림 6. 후보 모델 성능과 최종 모델의 실제값-예측값 비교')
rv=dict(zip(reporting['구분'],reporting['MAPE(%)'])); b3v=batch3_reporting['MAPE(%)'].tolist()
story += [PageBreak(),P('지정 성능 보고 포맷','H2K'),table(['구분','MAPE (%)','비고'],reporting_rows,[58,28,85]),
          P('Batch 3 보고 포맷은 위 6행에 아래 3행을 이어 붙인 형태입니다(outputs/day2_batch3_reporting.csv).','CaptionK'),
          table(['Batch 3 추가 검증','MAPE (%)','비고'],batch3_rows,[58,28,85]),
          P(f"Gap (Train-Valid)는 {rv['Gap (Train-Valid)']:+.2f}%p입니다. Valid가 {int(split['n_valid'])}셀뿐이라 불확실성이 커서 과적합 여부는 참고 수준으로만 판단합니다. Valid-Test Gap은 {rv['Gap (Valid-Test)']:+.2f}%p로 Batch 2 일반화 저하가 주요 위험이며, 원논문 Target 9.1% 대비 Batch 2 Gap은 {rv['Gap (Target-Test)']:+.2f}%p, Batch 3 Gap은 {b3v[-1]:+.2f}%p입니다. "
            f"Gap (Batch2-Batch3)가 {b3v[-2]:+.2f}%p로 큰 것은 Feature가 특정 배치에 과적합되었다기보다, Batch 2에 학습 범위보다 짧은 수명이 몰려 있기 때문입니다. 원논문에서도 제외한 Batch 3 노이즈 셀 4개를 빼면 Batch 3 MAPE는 {split['b3_clean_mape']:.2f}%({int(split['n_b3_clean'])}셀)입니다. "
            "원논문 9.1%는 더 많은 학습 셀과 원논문 기준 정제를 사용한 결과이고, 이 과제의 Batch 2에는 셀 구조가 다른 newstructure 셀 9개가 섞여 있어 조건이 완전히 같지는 않습니다.")]

story += [PageBreak(),P('8 결론과 한계','H1K'),P('세 Batch를 함께 분석한 결과, 배터리 수명은 초기 용량 상태뿐 아니라 충전 조건과 열 스트레스, 셀 구조가 결합된 결과로 해석하는 것이 적절합니다. Batch별 Target 분포가 다르므로 실제 적용 모델은 내부 검증 점수와 함께 독립 Batch 일반화 성능을 확인해야 합니다. 초기 100 Cycle의 log ΔQ 분산은 배치가 바뀌어도 유지되는 가장 안정적인 신호였고, 최종 모델은 학습 범위 안의 셀에서는 원논문과 비슷한 오차를 보였습니다.'),
          table(['한계','영향','운영 대응'],[['학습 범위 밖 외삽 실패','짧은·긴 수명 배치에서 오차 급증','새 배치 소량 라벨로 재보정, 예측 구간 함께 제공'],['Batch 2·3 Target 결측','평가 표본 감소','유효 셀 수와 제외 사유를 데이터 품질 지표로 관리'],['프로토콜별 표본 1~3셀','충전 조건 효과 불안정','충전정책별 표본을 확대하고 과해석 금지'],['공통 Knee point 없음','단일 Knee Feature 위험','Knee는 사후 설명용, 초기 Feature를 예측에 사용'],['Batch 2 과대예측 위험','교체 지연 가능성','보수적 안전계수와 수동 재검사 규칙 적용'],['Valid 8~10셀','분할에 따라 Valid 변동','반복 Hold-out 평균으로 모델 선택']],[42,47,77]),
          P('최종 모델은 Batch 1 내부 검증(CV·Hold-out)만으로 선택하고, Batch 2·3 Test MAPE, Valid-Test Gap, 과대예측 비율은 도입 여부와 운영 규칙을 정하는 데 사용합니다. 회사의 목적은 단순히 높은 점수를 얻는 것이 아니라, 배터리 교체 시점과 운영 위험을 더 일찍 판단할 수 있는 신뢰 가능한 기준을 확보하는 것입니다.')]

doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=16*mm,leftMargin=16*mm,topMargin=15*mm,bottomMargin=18*mm,title='ESS 배터리 수명 예측 DAY 1·DAY 2 보고서',author='')
doc.build(story,onFirstPage=cover_page,onLaterPages=footer)
print(PDF)
