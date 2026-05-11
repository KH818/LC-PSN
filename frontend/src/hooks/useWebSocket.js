import { useEffect, useState } from "react";

export function useWebSocket(url) {
  const [status, setStatus] = useState("disconnected");
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!url) return;

    const socket = new WebSocket(url);

    socket.onopen = () => {
      setStatus("connected");
      setError(null);
    };

    socket.onmessage = (event) => {
      try {
        setMessage(JSON.parse(event.data));
      } catch {
        setMessage(event.data);
      }
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

  return { status, message, error };
}