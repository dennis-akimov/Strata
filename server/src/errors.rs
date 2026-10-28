//! The server's failures, as the Python server's exception classes name them; the HTTP layer maps each to its status.

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Fail {
    /// ValueError: a request the server answers 400 (or, once a stream started, the engine's ERR line)
    #[error("{0}")]
    Value(String),
    /// EngineDied: the engine ended (or was ended) - 503 before an answer started
    #[error("{0}")]
    Died(String),
    /// EngineSilent (an EngineDied): the server ended an engine that lost step (#481)
    #[error("{0}")]
    Silent(String),
    /// EngineStarting: 503 while the engine starts
    #[error("{0}")]
    Starting(String),
    /// ModelBusy: 409
    #[error("{0}")]
    Busy(String),
    /// GpuBusy: 503
    #[error("{0}")]
    GpuBusy(String),
    /// EngineStuck: the engine would not end - 503
    #[error("{0}")]
    Stuck(String),
    /// SessionRefused: SERR <kind> <published> <reason>
    #[error("{message}")]
    Session { kind: String, message: String, published: bool },
    /// StructuredOutputError: 502
    #[error("{0}")]
    Structured(String),
    /// McpCancelled
    #[error("stopped")]
    McpCancelled,
}

impl Fail {
    /// isinstance(e, EngineDied)
    pub fn is_died(&self) -> bool {
        matches!(self, Fail::Died(_) | Fail::Silent(_))
    }
    /// isinstance(e, ValueError) (SessionRefused is one)
    pub fn is_value(&self) -> bool {
        matches!(self, Fail::Value(_) | Fail::Session { .. })
    }
    pub fn session_status(kind: &str) -> Option<u16> {
        Some(match kind {
            "invalid" => 400,
            "storage" => 507,
            "memory" => 503,
            "io" => 500,
            _ => return None,
        })
    }
}

pub fn value<T>(msg: impl Into<String>) -> Result<T, Fail> {
    Err(Fail::Value(msg.into()))
}

impl From<crate::frontend::BadRequest> for Fail {
    fn from(e: crate::frontend::BadRequest) -> Self {
        Fail::Value(e.0)
    }
}

impl From<crate::template::TemplateError> for Fail {
    fn from(e: crate::template::TemplateError) -> Self {
        Fail::Value(e.to_string())
    }
}
