# Think? Drink!☕️
개인별 적정 섭취량에 따른 음료 영양 기록 서비스입니다.

> **목적** 번거로운 개인별 영양성분 계산을 더욱 쉽고 편리하게 도우며, 마시고 싶은 음료의 영양성분을 한눈에 확인할 수 있는 서비스

> **기능** 카페 음료 이미지(스타벅스) 분류 모델 제공, 음료의 영양 성분 제공, 개인별 권장 섭취량 및 음료를 통한 섭취량 계산

> **구성원** 강재은, 박수정, 박진은, 이예주, 최예은

> **기간** 2022.06.27 ~ 2022.08.31  
* 사용한 언어 <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
* 모델 학습에 사용한 오픈소스 머신러닝 라이브러리 <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=Pytorch&logoColor=white">

  (2022년 프로젝트 진행 당시 사용한 머신러닝 라이브러리는 <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white"> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white">이며, 현재 레포지토리에 없습니다.)
* 모델 학습에 사용한 GPU 리소스 <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white">

## 개발 환경 및 구성 정보🔧
Python 3.8.19의 아나콘다 가상환경에서 진행되었으며, requirements.txt에서 패키지 버전에 맞추어 환경 설정 후 실행 바랍니다.  
_(Pytorch 모델 관련 패키지 정보만 제공됩니다.)_

## 프로젝트 내 역할
1. 크롤링을 이용하여 카페별 음료 영양 성분 데이터 수집
2. 스타벅스 음료 Top 10 (2022.08.XX 기준)의 이미지 수집 -> 이미지 데이터는 제공되지 않습니다.
3. 데이터셋 특징에 따른 이미지 분류 모델(CNN) 제작

📦Thinkdrink
 ┣ 📂data (❌미제공)
 ┣ 📂saved (❌모델 저장 폴더로 미제공)
 ┣ 📂source
 ┃ ┣ 📜dataset.py   #generate dataset
 ┃ ┣ 📜model.py   #model develop
 ┃ ┣ 📜setup.py   #initial arguments in model
 ┃ ┗ 📜trainer.py   #model trainer
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┣ 📜THINKDRINK팀_발표자료.pdf
 ┣ 📜main.py   #confirm train/evaluate model
 ┗ 📜requirements.txt