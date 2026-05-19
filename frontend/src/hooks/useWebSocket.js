import { useEffect, useRef, useState } from "react";
import { normalizeInferenceEvent } from "../utils/inferenceEvent";

const CACHE_KEY = "lcpsn:inference-cache:v1";
const MAX_HISTORY_LENGTH = 50;

function readInferenceCache() {
  try {
    const cached = window.localStorage.getItem(CACHE_KEY);
    if (!cached) return { latest: null, history: [], lastMessageAt: null };

    const parsed = JSON.parse(cached);

    return {
      latest: parsed.latest ?? null,
      history: Array.isArray(parsed.history) ? parsed.history.slice(-MAX_HISTORY_LENGTH) : [],
      lastMessageAt: parsed.lastMessageAt ?? null,
    };
  } catch {
    return { latest: null, history: [], lastMessageAt: null };
  }
}

function writeInferenceCache({ latest, history, lastMessageAt }) {
  try {
    window.localStorage.setItem(
      CACHE_KEY,
      JSON.stringify({
        version: 1,
        latest,
        history: history.slice(-MAX_HISTORY_LENGTH),
        lastMessageAt,
      })
    );
  } catch {
    // localStorage 저장이 실패해도 실시간 화면 갱신은 계속되어야 한다.
  }
}

function getEventKey(message) {
  const inference = normalizeInferenceEvent(message);

  // 백엔드가 event_id를 제공하면 같은 이벤트가 history에 중복 저장되지 않게 한다.
  return inference.eventId;
}

function appendMessage(history, nextMessage) {
  const nextKey = getEventKey(nextMessage);

  if (nextKey && history.some((message) => getEventKey(message) === nextKey)) {
    return history;
  }

  return [...history, nextMessage].slice(-MAX_HISTORY_LENGTH);
}

function parseSocketMessage(data) {
  try {
    return JSON.parse(data);
  } catch {
    return data;
  }
}

export function useWebSocket(url, { paused = false } = {}) {
  const [initialCache] = useState(readInferenceCache);
  const [status, setStatus] = useState("disconnected");
  const [message, setMessage] = useState(initialCache.latest);
  // Waterfall chart와 새로고침 복구를 위해 최근 추론 이벤트를 제한된 개수만 저장한다.
  const [messageHistory, setMessageHistory] = useState(initialCache.history);
  const [lastMessageAt, setLastMessageAt] = useState(initialCache.lastMessageAt);
  const [error, setError] = useState(null);

  const latestRef = useRef(initialCache.latest);
  const historyRef = useRef(initialCache.history);
  const lastMessageAtRef = useRef(initialCache.lastMessageAt);
  const pausedRef = useRef(paused);

  useEffect(() => {
    pausedRef.current = paused;

    if (!paused) {
      setMessage(latestRef.current);
      setMessageHistory(historyRef.current);
      setLastMessageAt(lastMessageAtRef.current);
    }
  }, [paused]);

  useEffect(() => {
    if (!url) return;

    const socket = new WebSocket(url);

    socket.onopen = () => {
      setStatus("connected");
      setError(null);
    };

    socket.onmessage = (event) => {
      // 백엔드가 JSON 문자열을 보내면 객체로 변환하고, 실패하면 원문 문자열을 유지한다.
      const nextMessage = parseSocketMessage(event.data);
      const receivedAt = new Date().toISOString();
      const nextHistory = appendMessage(historyRef.current, nextMessage);

      latestRef.current = nextMessage;
      historyRef.current = nextHistory;
      lastMessageAtRef.current = receivedAt;

      writeInferenceCache({
        latest: nextMessage,
        history: nextHistory,
        lastMessageAt: receivedAt,
      });

      if (pausedRef.current) {
        return;
      }

      // pause 상태가 아니면 최신 메시지를 즉시 화면에 반영한다.
      setMessage(nextMessage);
      setMessageHistory(nextHistory);
      setLastMessageAt(receivedAt);
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
