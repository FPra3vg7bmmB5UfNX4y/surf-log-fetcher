update public.wind_forecast_history set source = 'open-meteo' where source is null;
alter table public.wind_forecast_history alter column source set not null;
create unique index if not exists wind_forecast_history_source_dedup
  on public.wind_forecast_history (source, issued_at, valid_at);
alter table public.wind_forecast_history
  drop constraint if exists wind_forecast_history_dedup;

alter table public.conditions
  add column if not exists wind_obs_at timestamptz;
