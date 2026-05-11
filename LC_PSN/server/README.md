# LC-PSN FastAPI Mock Inference Server

## 1. 개요

본 서버는 LC-PSN / TransMUSIC 기반 DOA 추정 모델을 실제 서비스 형태로 사용하기 위한 FastAPI 기반 추론 서버의 초기 뼈대이다.

1주차 구현 범위에서는 실제 PyTorch 모델을 연결하지 않고, Mock 추론 결과를 생성하여 다음 흐름을 검증한다.

```text
클라이언트 요청
→ FastAPI 서버 수신
→ Mock DOA 추론 결과 생성
→ REST API 응답 반환
→ WebSocket으로 실시간 결과 송출

## 2. 서버 실행 방법
서버 실행 명령어: python -m uvicorn server.main:app --reload
서버 기본 주소: http://127.0.0.1:8000 -> 서버 실행 여부 확인
Health check: http://127.0.0.1:8000/health -> 서버 상태 확인
API 문서 테스트: http://127.0.0.1:8000/docs -> Swagger UI에서 API 테스트
Websocket 테스트: ws://127.0.0.1:8000/ws -> 실시간 결과 수신용 웹소켓 주소
