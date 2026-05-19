import { useEffect, useState } from "react";

export function useWebSocket(url) {
  const [status, setStatus] = useState("disconnected");
  const [message, setMessage] = useState(null);
  // Waterfall chart에서 최근 프레임을 누적해서 보여주기 위한 메시지 히스토리
  const [messageHistory, setMessageHistory] = useState([]);
  const [lastMessageAt, setLastMessageAt] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!url) return;

    const socket = new WebSocket(url);

    socket.onopen = () => {
      setStatus("connected");
      setError(null);
    };

    socket.onmessage = (event) => {
      let nextMessage = event.data;

      // 백엔드가 JSON 문자열을 보내면 객체로 변환하고, 실패하면 원문 문자열을 유지한다.
      try {
        nextMessage = JSON.parse(event.data);
      } catch {
        nextMessage = event.data;
      }

      setMessage(nextMessage);
      // 화면 부담을 줄이기 위해 최근 36개 프레임만 유지한다.
      setMessageHistory((current) => [...current.slice(-35), nextMessage]);
      setLastMessageAt(new Date().toISOString());
    };

    socket.onerror = () => {
      setStatus("error");
      setError("WebSocket connection error");
    };

    socket.onclose = () => {
      setStatus("disconnected");
    };

    return () => {
      socket.close();
    };
  }, [url]);

  return { status, message, messageHistory, lastMessageAt, error };
}
