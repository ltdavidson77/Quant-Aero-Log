-- Database initialization script for Quant-Aero-Log

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS quantaerolog;
CREATE SCHEMA IF NOT EXISTS metrics;

-- Set search path
SET search_path TO quantaerolog, public;

-- Create user roles
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'quantaerolog_read') THEN
        CREATE ROLE quantaerolog_read;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'quantaerolog_write') THEN
        CREATE ROLE quantaerolog_write;
    END IF;
END
$$;

-- Create tables
CREATE TABLE IF NOT EXISTS market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(18,8) NOT NULL,
    high DECIMAL(18,8) NOT NULL,
    low DECIMAL(18,8) NOT NULL,
    close DECIMAL(18,8) NOT NULL,
    volume DECIMAL(18,8) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_data_id UUID NOT NULL REFERENCES market_data(id),
    signal_type VARCHAR(50) NOT NULL,
    value DECIMAL(18,8) NOT NULL,
    confidence DECIMAL(5,4),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID NOT NULL REFERENCES signals(id),
    model_version VARCHAR(50) NOT NULL,
    prediction DECIMAL(18,8) NOT NULL,
    probability DECIMAL(5,4),
    horizon_minutes INTEGER NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(18,8) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create hypertables
SELECT create_hypertable('market_data', 'timestamp');
SELECT create_hypertable('signals', 'created_at');
SELECT create_hypertable('predictions', 'created_at');
SELECT create_hypertable('model_metrics', 'timestamp');

-- Create indexes
CREATE INDEX idx_market_data_symbol ON market_data(symbol);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp DESC);
CREATE INDEX idx_signals_market_data_id ON signals(market_data_id);
CREATE INDEX idx_signals_type ON signals(signal_type);
CREATE INDEX idx_predictions_signal_id ON predictions(signal_id);
CREATE INDEX idx_predictions_model ON predictions(model_version);
CREATE INDEX idx_model_metrics_version ON model_metrics(model_version);

-- Create updated_at triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_market_data_updated_at
    BEFORE UPDATE ON market_data
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_signals_updated_at
    BEFORE UPDATE ON signals
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_predictions_updated_at
    BEFORE UPDATE ON predictions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT USAGE ON SCHEMA quantaerolog TO quantaerolog_read, quantaerolog_write;
GRANT SELECT ON ALL TABLES IN SCHEMA quantaerolog TO quantaerolog_read;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA quantaerolog TO quantaerolog_write;
ALTER DEFAULT PRIVILEGES IN SCHEMA quantaerolog GRANT SELECT ON TABLES TO quantaerolog_read;
ALTER DEFAULT PRIVILEGES IN SCHEMA quantaerolog GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO quantaerolog_write;

-- Create metrics schema tables
SET search_path TO metrics, public;

CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(18,8) NOT NULL,
    labels JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS application_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(18,8) NOT NULL,
    labels JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create hypertables for metrics
SELECT create_hypertable('system_metrics', 'timestamp');
SELECT create_hypertable('application_metrics', 'timestamp');

-- Create indexes for metrics
CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp DESC);
CREATE INDEX idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX idx_application_metrics_timestamp ON application_metrics(timestamp DESC);
CREATE INDEX idx_application_metrics_name ON application_metrics(metric_name);

-- Grant permissions for metrics schema
GRANT USAGE ON SCHEMA metrics TO quantaerolog_read, quantaerolog_write;
GRANT SELECT ON ALL TABLES IN SCHEMA metrics TO quantaerolog_read;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA metrics TO quantaerolog_write;
ALTER DEFAULT PRIVILEGES IN SCHEMA metrics GRANT SELECT ON TABLES TO quantaerolog_read;
ALTER DEFAULT PRIVILEGES IN SCHEMA metrics GRANT SELECT, INSERT ON TABLES TO quantaerolog_write; 