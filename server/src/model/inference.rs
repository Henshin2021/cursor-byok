use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{ModelSpec, ProjectedContent, ProjectedMessage, ToolDefinition};

const PROVIDER_TOOL_CALL_ID_MAX_BYTES: usize = 64;
const PROVIDER_TOOL_CALL_ID_DIGEST_BYTES: usize = 12;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PromptSpec {
    pub instructions: String,
    pub tools: Vec<ToolDefinition>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelRequest {
    pub prompt: PromptSpec,
    pub model: ModelSpec,
    pub history: Vec<ProjectedMessage>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelInvocation {
    pub call_id: String,
    pub run_id: String,
    pub conversation_id: String,
    pub provider_call_index: u64,
    pub request: ModelRequest,
}

pub(crate) fn normalize_provider_tool_call_ids(history: &mut [ProjectedMessage]) {
    for message in history {
        match &mut message.content {
            ProjectedContent::Assistant { calls, .. } => {
                for call in calls {
                    shorten_tool_call_id(&mut call.call_id);
                }
            }
            ProjectedContent::ToolResult(result) => {
                shorten_tool_call_id(&mut result.call_id);
            }
            ProjectedContent::Parts(_) => {}
        }
    }
}

fn shorten_tool_call_id(call_id: &mut String) {
    if call_id.len() <= PROVIDER_TOOL_CALL_ID_MAX_BYTES {
        return;
    }
    let digest = Sha256::digest(call_id.as_bytes());
    *call_id = format!(
        "call_{}",
        hex::encode(&digest[..PROVIDER_TOOL_CALL_ID_DIGEST_BYTES])
    );
}

#[cfg(test)]
mod tests {
    use super::{normalize_provider_tool_call_ids, PROVIDER_TOOL_CALL_ID_MAX_BYTES};
    use crate::model::{
        ProjectedContent, ProjectedMessage, Role, ToolCallContent, ToolResultContent,
    };

    #[test]
    fn long_provider_tool_call_ids_map_to_one_short_id_across_history() {
        let call_id = format!("cursor-tool-call:{}", "x".repeat(80));
        let mut history = vec![
            assistant_with_call(&call_id),
            ProjectedMessage {
                message_id: "result".into(),
                role: Role::Tool,
                content: ProjectedContent::ToolResult(ToolResultContent {
                    call_id: call_id.clone(),
                    name: "Shell".into(),
                    content: "done".into(),
                    is_error: false,
                    image: None,
                    provider_parts: Vec::new(),
                }),
            },
        ];

        normalize_provider_tool_call_ids(&mut history);

        let shortened = first_call_id(&history[0]);
        let ProjectedContent::ToolResult(result) = &history[1].content else {
            panic!("expected tool result");
        };
        assert_eq!(result.call_id, shortened);
        assert_eq!(shortened.len(), 29);
        assert!(shortened.starts_with("call_"));
        assert!(shortened.is_ascii());
    }

    #[test]
    fn ids_within_the_limit_are_left_untouched() {
        let call_id = "x".repeat(PROVIDER_TOOL_CALL_ID_MAX_BYTES);
        let mut history = vec![assistant_with_call(&call_id)];

        normalize_provider_tool_call_ids(&mut history);

        assert_eq!(first_call_id(&history[0]), call_id);
    }

    #[test]
    fn ids_sharing_a_long_prefix_stay_distinct() {
        let prefix = "x".repeat(80);
        let mut history = vec![
            assistant_with_call(&format!("{prefix}a")),
            assistant_with_call(&format!("{prefix}b")),
        ];

        normalize_provider_tool_call_ids(&mut history);

        assert_ne!(first_call_id(&history[0]), first_call_id(&history[1]));
    }

    #[test]
    fn multi_byte_ids_are_measured_in_bytes_and_become_ascii() {
        let call_id = "界".repeat(30);
        assert!(call_id.chars().count() < PROVIDER_TOOL_CALL_ID_MAX_BYTES);
        assert!(call_id.len() > PROVIDER_TOOL_CALL_ID_MAX_BYTES);
        let mut history = vec![assistant_with_call(&call_id)];

        normalize_provider_tool_call_ids(&mut history);

        let shortened = first_call_id(&history[0]);
        assert!(shortened.is_ascii());
        assert!(shortened.len() <= PROVIDER_TOOL_CALL_ID_MAX_BYTES);
    }

    fn assistant_with_call(call_id: &str) -> ProjectedMessage {
        ProjectedMessage {
            message_id: format!("assistant-{call_id}"),
            role: Role::Assistant,
            content: ProjectedContent::Assistant {
                text: String::new(),
                thinking: String::new(),
                replay_state: None,
                calls: vec![ToolCallContent {
                    index: 0,
                    call_id: call_id.into(),
                    name: "Shell".into(),
                    arguments: serde_json::json!({}),
                }],
            },
        }
    }

    fn first_call_id(message: &ProjectedMessage) -> String {
        let ProjectedContent::Assistant { calls, .. } = &message.content else {
            panic!("expected assistant message");
        };
        calls[0].call_id.clone()
    }
}
