# Image_Stitching

1. 환경설치
파이썬 버전 3.8.11설치 후 아래 명령어를 통해 설치
'''
pip install -r requirements.txt
'''

* 실패시 pytorch의 이슈일 경우가 높음. 아래 링크를 참고하여 pytorch1.8.0 버전을 설치하고,
* https://pytorch.org/get-started/previous-versions/ 
* requirements.txt의 윗 두줄을 지운 뒤 다시 pip install -r requirements.txt

2. api 작동법
환경설정 후 아래명령어로 작동
python main.py --port 원하는 포트번호(숫자)

3. 개발코드
개발 코드는 패키지화하여 stitching폴더에 있음.
각 기능별로 주석 및 하단에 예시 코드 작성되어 있음.

4. 인공지능 모델
stitching/saved_models에 ~~저장되어있음.~~ 현 레파지토리에선 제공하지 않음.
