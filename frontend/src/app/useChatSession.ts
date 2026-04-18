import { useEffect, useRef, useState } from "react";
import type {
  ChatHistorySnapshot,
  ChatMessage,
  HistorySummary,
  ModelOptionValue,
} from "../types";
import { formatTimestamp, parseSseBlock } from "./utils";

interface UseChatSessionOptions {
  apiBaseUrl: string;
  model: ModelOptionValue;
  refreshHistoryList: () => void | Promise<void>;
  selectHistory: (historyId: number) => void;
  upsertHistoryItem: (summary: HistorySummary) => void;
}

interface ResetChatStateOptions {
  abort?: boolean;
  clearActiveHistoryId?: boolean;
}

export function useChatSession({
  apiBaseUrl,
  model,
  refreshHistoryList,
  selectHistory,
  upsertHistoryItem,
}: UseChatSessionOptions) {
  const [activeChatHistoryId, setActiveChatHistoryId] = useState<number | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatThinking, setChatThinking] = useState(false);
  const [chatReasoningStreaming, setChatReasoningStreaming] = useState("");
  const [chatStreamingText, setChatStreamingText] = useState("");
  const [chatError, setChatError] = useState("");

  const chatAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      chatAbortRef.current?.abort();
    };
  }, []);

  function resetChatState(options: ResetChatStateOptions = {}) {
    if (options.abort) {
      chatAbortRef.current?.abort();
      chatAbortRef.current = null;
    }
    if (options.clearActiveHistoryId !== false) {
      setActiveChatHistoryId(null);
    }
    setChatMessages([]);
    setChatInput("");
    setChatThinking(false);
    setChatReasoningStreaming("");
    setChatStreamingText("");
    setChatError("");
  }

  function clearActiveChatHistoryId() {
    setActiveChatHistoryId(null);
  }

  async function handleChatSend() {
    const nextMessage = chatInput.trim();
    if (!nextMessage || chatThinking) {
      return;
    }

    const userMessage: ChatMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      content: nextMessage,
      at: formatTimestamp(),
    };

    setChatMessages((current) => [...current, userMessage]);
    setChatInput("");
    setChatThinking(true);
    setChatReasoningStreaming("");
    setChatStreamingText("");
    setChatError("");

    chatAbortRef.current?.abort();
    const controller = new AbortController();
    chatAbortRef.current = controller;

    const requestMessages = [...chatMessages, userMessage].map((message) => ({
      role: message.role,
      content: message.content,
      at: message.at,
    }));

    try {
      const response = await fetch(`${apiBaseUrl}/api/chat/stream`, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: requestMessages,
          ...(model ? { model } : {}),
          history_id: activeChatHistoryId,
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const payload = await response.text();
        throw new Error(payload || `HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalMessage = "";
      let finalReasoning = "";
      let returnedHistoryId: number | null = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";

        for (const block of blocks) {
          if (!block.trim()) {
            continue;
          }
          const parsed = parseSseBlock(block);
          if (parsed.event === "chat_chunk") {
            const chunk = String(parsed.data.chunk ?? "");
            if (chunk) {
              finalMessage += chunk;
              setChatStreamingText((current) => current + chunk);
            }
          } else if (parsed.event === "chat_reasoning_chunk") {
            const chunk = String(parsed.data.chunk ?? "");
            if (chunk) {
              finalReasoning += chunk;
              setChatReasoningStreaming((current) => current + chunk);
            }
          } else if (parsed.event === "result") {
            finalMessage = String(parsed.data.message ?? finalMessage);
            finalReasoning = String(parsed.data.reasoning ?? finalReasoning);
            if (typeof parsed.data.history_id === "number") {
              returnedHistoryId = Number(parsed.data.history_id);
            }
            setChatStreamingText(finalMessage);
            setChatReasoningStreaming(finalReasoning);
          } else if (parsed.event === "error") {
            throw new Error(String(parsed.data.message ?? "Chat request failed."));
          }
        }
      }

      if (buffer.trim()) {
        const parsed = parseSseBlock(buffer);
        if (parsed.event === "chat_chunk") {
          const chunk = String(parsed.data.chunk ?? "");
          if (chunk) {
            finalMessage += chunk;
            setChatStreamingText((current) => current + chunk);
          }
        } else if (parsed.event === "chat_reasoning_chunk") {
          const chunk = String(parsed.data.chunk ?? "");
          if (chunk) {
            finalReasoning += chunk;
            setChatReasoningStreaming((current) => current + chunk);
          }
        } else if (parsed.event === "result") {
          finalMessage = String(parsed.data.message ?? finalMessage);
          finalReasoning = String(parsed.data.reasoning ?? finalReasoning);
          if (typeof parsed.data.history_id === "number") {
            returnedHistoryId = Number(parsed.data.history_id);
          }
          setChatStreamingText(finalMessage);
          setChatReasoningStreaming(finalReasoning);
        } else if (parsed.event === "error") {
          throw new Error(String(parsed.data.message ?? "Chat request failed."));
        }
      }

      const cleanedMessage = finalMessage.trim();
      if (!cleanedMessage) {
        throw new Error("Chat model returned an empty response.");
      }

      const assistantMessage: ChatMessage = {
        id: `${Date.now()}-assistant`,
        role: "assistant",
        content: cleanedMessage,
        reasoning: finalReasoning.trim() || undefined,
        at: formatTimestamp(),
      };
      setChatMessages((current) => [...current, assistantMessage]);
      setChatReasoningStreaming("");
      setChatStreamingText("");

      if (returnedHistoryId) {
        setActiveChatHistoryId(returnedHistoryId);
        selectHistory(returnedHistoryId);
        upsertHistoryItem({
          id: returnedHistoryId,
          mode: "chat",
          title: nextMessage.slice(0, 80),
          preview_text: cleanedMessage.slice(0, 120),
          model,
          language: null,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
      }
      void refreshHistoryList();
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }
      setChatError(error instanceof Error ? error.message : String(error));
      setChatReasoningStreaming("");
      setChatStreamingText("");
    } finally {
      setChatThinking(false);
      if (chatAbortRef.current === controller) {
        chatAbortRef.current = null;
      }
    }
  }

  function loadHistorySnapshot(historyId: number, snapshot: ChatHistorySnapshot) {
    setActiveChatHistoryId(historyId);
    setChatMessages(
      (snapshot.messages ?? []).map((message, index) => ({
        id: `history-${historyId}-${index}`,
        role: message.role,
        content: message.content,
        reasoning: typeof message.reasoning === "string" ? message.reasoning : undefined,
        at: message.at,
      })),
    );
    setChatInput("");
    setChatThinking(false);
    setChatReasoningStreaming("");
    setChatStreamingText("");
    setChatError("");
  }

  return {
    activeChatHistoryId,
    chatMessages,
    chatInput,
    chatThinking,
    chatReasoningStreaming,
    chatStreamingText,
    chatError,
    setChatInput,
    setChatError,
    clearActiveChatHistoryId,
    resetChatState,
    handleChatSend,
    loadHistorySnapshot,
  };
}
