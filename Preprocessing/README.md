# Preprocessing

### 1. F를 제외한 나머지 데이터 삭제
### 2. 불필요한 구간(앞, 뒤) 삭제 - 영상(mp4 -> frame), json
- 형태소(json)로부터 자르는 구간 정보 수집
- 영상 편집 (1. 필요한 구간에 대해서만 2. 프레임 변환)
- Json 편집 (필요한 구간을 제외한 json파일 삭제)
### 3. Keypoints 전처리
- pose, face, hand 다 1차원 리스트로 연결
- confidence 값 제외
- pose에서 하반신 0 제외
- 그 외의 0 값 앞 뒤 평균으로 채우기
- 0~1 scaling
