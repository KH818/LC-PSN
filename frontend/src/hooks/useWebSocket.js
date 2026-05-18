import { useEffect, useState } from "react";

export function useWebSocket(url) {
  const [status, setStatus] = useState("disconnected");
  const [message, setMessage] = useState(null);
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

      try {
        nextMessage = JSON.parse(event.data);
      } catch {
        nextMessage = event.data;
      }

      setMessage(nextMessage);
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
