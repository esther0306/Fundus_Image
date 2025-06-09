# 👁️ 광각 안저 영상 기반 안질환 다중 분류 및 Grad-CAM 시각화  
**Wide-field Fundus Imaging Eye Disease Multiclassification and Grad-CAM Visualization**
> ⚠️ 안심존에서는 외부 API, 프레임워크 가중치 다운로드 불가 → Pretrained 사용 제한
---

## 📌 프로젝트 개요

본 프로젝트는 **광각 안저(fundus) 이미지**를 기반으로  
정상과 세 가지 주요 실명 유발 안질환을 분류하는 **딥러닝 기반 다중 분류 모델**을 개발하는 것을 목표로 합니다.

- **대상 질환**:
  - 당뇨망막병증 *(Diabetic Retinopathy)*
  - 황반변성 *(Macular Degeneration)*
  - 녹내장 *(Glaucoma)*

모델은 **ResNet-18** 기반의 구조로 사전 학습 없이 구성되었으며,  
**양안(좌우 눈) 정보 통합**과 **Grad-CAM 시각화**를 통해 임상적 신뢰성을 확보하였습니다.

> 🎯 실험 결과: **정확도 69.68%**  
> 📌 향후: 전이 학습 및 구조 기반 전처리를 통한 성능 개선 기대

---

## 🔍 연구 배경

- 고령화로 인해 **실명을 유발하는 안질환의 유병률 증가**
- 40세 이상 국내 유병률:
  - 당뇨망막병증 19.6%
  - 황반변성 13.4%
  - 녹내장 3.4% (질병관리청 / 대한안과학회)
- **안저 검사는 비침습적으로 망막과 시신경을 확인**할 수 있는 핵심 진단 도구
- 최근에는 **딥러닝 기술을 통한 안저 영상 기반 질환 자동 분류** 연구가 활발히 진행 중

---

## 🧠 연구 목적

- 광각 안저 이미지를 활용한 **다중 질환 분류 모델 구축**
- **Grad-CAM 시각화**를 통해 모델이 실제로 질환 관련 영역에 주목하는지 분석
- 임상에서 활용 가능한 **설명 가능 인공지능(XAI)** 기반 진단 도구 구축

---

## 🧾 요약 정보

| 항목 | 내용 |
|------|------|
| 모델 | ResNet-18 기반 CNN |
| 분류 대상 | 정상 / 당뇨망막병증 / 황반변성 / 녹내장 |
| 시각화 기법 | Grad-CAM |
| 주요 기여 | 양안 이미지 통합, 비사전학습 모델, 시각화 통한 임상 해석 가능성 |
| 정확도 | 69.68% |
| 향후 계획 | Transfer Learning, 구조기반 전처리, 성능 향상

---

## 🔑 키워드

`Fundus Image` · `Deep Learning` · `ResNet-18` · `Grad-CAM` · `Multi-disease Classification`

## 🗂️ 데이터셋 구성

본 연구에서는 **건양대학교병원에서 수집된 광각 안저 영상 데이터**를 기반으로  
정상 및 3가지 주요 안질환(황반변성, 당뇨망막병증, 망막정맥폐쇄)을 포함한 총 4개 클래스를 분류하였습니다.

- 총 데이터 수: **2,202장**
- 소스: **AI허브** 제공 '질병진단 이미지(안저)' 학습용 데이터셋 [NIA 공개 링크[5]]  
- 모든 영상은 **전공의의 1차 판독 + 안과 전문의의 2차 검수**를 통해 라벨링

| 클래스         | 이미지 수 |
|----------------|-----------|
| 정상           | 613장     |
| 황반변성       | 502장     |
| 당뇨망막병증   | 853장     |
| 망막정맥폐쇄   | 234장     |
| **합계**        | **2,202장** |

### 📷 이미지 예시  
<img src="https://github.com/user-attachments/assets/5aa1e417-b2a7-4c35-b150-3b4f740ccc74" width="400"/>

---

## ⚠️ 환경 및 제약 조건

본 프로젝트는 **안심존 보안 환경**에서 수행되었으며, 다음과 같은 제한 하에 연구가 진행되었습니다:

- 외부 인터넷 접속 불가 → **사전학습 가중치 사용 금지**
- 외부 모델 및 API 활용 불가 → **사전 전이 학습 없이 from scratch 학습**
- 제한된 GPU 자원 → **경량 모델(ResNet18) 기반 구현**
- 민감 의료정보 → **데이터 증강 미적용**, 원본 이미지 기반 학습

---

## 🛠 데이터 전처리

- 모든 이미지를 **224×224 픽셀로 리사이즈**
- 전체 데이터를 `train : val : test = 8 : 1 : 1`로 분할
- **데이터 증강은 미적용**, 원본 영상만 사용하여 모델 학습

> 🎯 의료 영상의 신뢰성과 정합성 확보를 위해, **클린 데이터 기반 정직한 학습 전략**을 채택했습니다.

##  모델 구성 및 학습 전략

본 연구의 전체 분류 파이프라인은 아래와 같습니다.

<img src="https://github.com/user-attachments/assets/8726b352-13a6-489c-9dac-05faf19cbc40" width="600"/>

- **Baseline 모델**: 가벼운 Sequential 구조 CNN
- **본 모델**: ResNet-18 기반 분류기  
- 사전학습(pretrained) 모델 사용은 계획되었으나, **안심존 보안 환경으로 인해 외부 가중치 불가**
- 따라서 **모델을 처음부터 구성하여 학습(from scratch)**
- 모든 모델은 동일한 하이퍼파라미터로 학습하여 비교의 공정성 확보


---

## 🔍 Grad-CAM 시각화

모델이 안저 영상 내 어떤 부분을 기준으로 판단했는지 시각적으로 확인하기 위해  
**Grad-CAM (Gradient-weighted Class Activation Mapping)** 기법을 사용했습니다.

- **적용 대상**: 가장 높은 분류 정확도를 보인 모델
- **Grad-CAM 특징**:
  - 기존 CAM은 GAP(Gobal Average Pooling) 이후 구조에 한정됨
  - **Grad-CAM은 Conv Layer의 Gradient를 기반**으로 Heatmap 생성
  - 다양한 구조에서 활용 가능하며, 임상의가 직관적으로 결과를 해석할 수 있게 도움

> Grad-CAM을 통해 딥러닝 모델이 실제 질환 관련 영역에 주목했는지를 검증할 수 있었으며,  
> 모델 해석 가능성과 임상적 활용 가능성을 높이는 데 기여하였습니다.

## 📊 실험 결과

### 🔧 학습 세팅
- **프레임워크**: TensorFlow
- **모델**: ResNet-18 (from scratch)
- **Optimizer**: Adam
- **Learning Rate**: 0.01
- **Scheduler**: Cosine Annealing (40 Epochs)
- **Loss Function**: Cross Entropy Loss
- **Metric**: Accuracy

### 📈 학습 정확도 및 Confusion Matrix

<img src="https://github.com/user-attachments/assets/9ddaa13a-819d-4a4f-b5f5-805fd0df3650" width="420"/>
<img src="https://github.com/user-attachments/assets/31369d64-9559-427d-aa04-9e48823375e1" width="320"/>

- 최종 테스트 정확도 (**Accuracy**)는 **69.68%** 기록
- Confusion Matrix를 통해 클래스별 예측 정확도 분포 확인 가능
- 모델 학습은 점진적으로 안정되었고, 클래스 간 구분 성능도 고르게 향상됨

---

###  기존 연구와의 비교

| 학습 방법            | 데이터 수 | AUC     | Accuracy |
|---------------------|-----------|---------|----------|
| Honggu Kang et al. [6] | 10,000    | **0.8955** | N/A      |
| Sequential (Baseline)  | 2,202     | N/A     | 54.5%    |
| **ResNet-18 (Ours)**   | 2,202     | N/A     | **69.68%** |

> 본 연구는 AUC 기반 성능을 직접 비교하기는 어렵지만, 소량의 학습 데이터에서도 신뢰성 있는 정확도를 달성하였음.

---

##  Grad-CAM 시각화 결과

- Grad-CAM을 통해 **모델이 실제 질환 분류 시 주목한 영역**을 시각적으로 확인
- 주요 병변 부위인 **시신경 유두, 망막 중심부 등**에 집중되어 있는 heatmap 확인
- 모델의 판단 기준이 임상 지식과 일치함을 확인 → **설명 가능성 및 신뢰성 향상**

---

## ✅ 결론

- 본 연구는 광각 안저 이미지를 활용하여 **다중 안질환 분류 모델**을 구축함
- 전처리 없이도 시신경 유두, 망막혈관 등 중요한 구조를 활용할 수 있었음
- Grad-CAM을 통해 모델이 **임상적으로 유의미한 영역**을 기반으로 판단함을 확인
- 향후에는 다음을 통해 성능 향상이 기대됨:
  - 명시적인 임상 구조 강조 전처리
  - 사전학습(pretrained) 모델 도입
  - 외부 데이터셋 확장

---

## 📚 참고문헌

[1] J. Chen et al., *Global impact of population aging on vision loss prevalence*, Global Transitions, 2024  
[2] 한국망막학회, https://www.retina.or.kr  
[3] 질병관리청, https://www.kdca.go.kr  
[4] R. Selvaraju et al., *Grad-CAM*, ICCV, 2017  
[5] AI Hub, https://www.aihub.or.kr  
[6] K. He et al., *Deep Residual Learning*, CVPR, 2016

















