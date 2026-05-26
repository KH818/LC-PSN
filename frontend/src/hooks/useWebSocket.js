import { useCallback, useEffect, useRef, useState } from "react";
import { normalizeInferenceEvent } from "../utils/inferenceEvent";

const CACHE_KEY = "lcpsn:inference-cache:v1";
const MAX_HISTORY_LENGTH = 50;
const MAX_RECONNECT_DELAY_MS = 8000;
const RENDER_THROTTLE_MS = 500;

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

function getReconnectDelay(attempt) {
  return Math.min(1000 * 2 ** Math.max(0, attempt - 1), MAX_RECONNECT_DELAY_MS);
}

export function useWebSocket(url, { paused = false } = {}) {
  const [initialCache] = useState(readInferenceCache);
  const [status, setStatus] = useState("disconnected");
  const [message, setMessage] = useState(initialCache.latest);
  // Waterfall chart와 새로고침 복구를 위해 최근 추론 이벤트를 제한된 개수만 저장한다.
  const [messageHistory, setMessageHistory] = useState(initialCache.history);
  const [lastMessageAt, setLastMessageAt] = useState(initialCache.lastMessageAt);
  const [error, setError] = useState(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);

  const latestRef = useRef(initialCache.latest);
  const historyRef = useRef(initialCache.history);
  const lastMessageAtRef = useRef(initialCache.lastMessageAt);
  const pausedRef = useRef(paused);
  const pendingRenderRef = useRef(null);
  const renderTimerRef = useRef(null);
  const lastRenderAtRef = useRef(0);

  const flushPendingRender = useCallback(() => {
    const pending = pendingRenderRef.current;
    if (!pending) return;

    pendingRenderRef.current = null;
    renderTimerRef.current = null;
    lastRenderAtRef.current = Date.now();
    setMessage(pending.message);
    setMessageHistory(pending.history);
    setLastMessageAt(pending.lastMessageAt);
  }, []);

  const scheduleRender = useCallback((nextMessage, nextHistory, receivedAt) => {
    pendingRenderRef.current = {
      message: nextMessage,
      history: nextHistory,
      lastMessageAt: receivedAt,
    };

    const elapsed = Date.now() - lastRenderAtRef.current;

    if (elapsed >= RENDER_THROTTLE_MS) {
      if (renderTimerRef.current) {
        window.clearTimeout(renderTimerRef.current);
      }

      flushPendingRender();
      return;
    }

    if (!renderTimerRef.current) {
      renderTimerRef.current = window.setTimeout(flushPendingRender, RENDER_THROTTLE_MS - elapsed);
    }
  }, [flushPendingRender]);

  useEffect(() => {
    pausedRef.current = paused;

    if (!paused) {
      pendingRenderRef.current = null;

      if (renderTimerRef.current) {
        window.clearTimeout(renderTimerRef.current);
        renderTimerRef.current = null;
      }

      setMessage(latestRef.current);
      setMessageHistory(historyRef.current);
      setLastMessageAt(lastMessageAtRef.current);
      lastRenderAtRef.current = Date.now();
    }
  }, [paused]);

  useEffect(() => {
    if (!url) return undefined;

    let socket = null;
    let reconnectTimerId = null;
    let closedByEffect = false;
    let reconnectCount = 0;

    const clearReconnectTimer = () => {
      if (reconnectTimerId) {
        window.clearTimeout(reconnectTimerId);
        reconnectTimerId = null;
      }
    };

    const scheduleReconnect = () => {
      if (closedByEffect) return;

      reconnectCount += 1;
      const delay = getReconnectDelay(reconnectCount);

      setStatus("reconnecting");
      setReconnectAttempt(reconnectCount);
      reconnectTimerId = window.setTimeout(() => {
        connect();
      }, delay);
    };

    const connect = () => {
      clearReconnectTimer();
      socket = new WebSocket(url);

      socket.onopen = () => {
        reconnectCount = 0;
        setStatus("connected");
        setReconnectAttempt(0);
        setError(null);
      };

      socket.onmessage = (event) => {
        // 수신과 캐시는 즉시 처리하고, 화면 렌더링만 throttle로 제한한다.
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

        scheduleRender(nextMessage, nextHistory, receivedAt);
      };

      socket.onerror = () => {
        setStatus("error");
        setError("WebSocket connection error");
      };

      socket.onclose = () => {
        if (closedByEffect) return;

        setStatus("disconnected");
        scheduleReconnect();
      };
    };

    connect();

    return () => {
      closedByEffect = true;
      clearReconnectTimer();

      if (renderTimerRef.current) {
        window.clearTimeout(renderTimerRef.current);
        renderTimerRef.current = null;
      }

      if (socket) {
        socket.close();
      }
    };
  }, [scheduleRender, url]);

  return { status, message, messageHistory, lastMessageAt, reconnectAttempt, error };
}
