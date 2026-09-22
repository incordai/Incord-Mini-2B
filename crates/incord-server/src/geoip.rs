//! Offline GeoIP lookup using MaxMind GeoLite2 database.
//!
//! At startup, the server may load a .mmdb file (downloaded from MaxMind —
//! free signup at https://www.maxmind.com/en/geolite2/signup). If the file
//! is absent or fails to load, all lookups gracefully return None — the server
//! continues normally and only misses geo enrichment.
//!
//! In tests, GeoDb::empty() is always used (no .mmdb file needed).

use std::net::IpAddr;
use std::path::Path;
use std::sync::Arc;

use maxminddb::Reader;

/// Loaded GeoIP database handle. Cheap to clone (Arc inside).
#[derive(Clone)]
pub struct GeoDb {
    inner: Option<Arc<Reader<Vec<u8>>>>,
}

/// Result of a GeoIP lookup — all fields are optional because any may be
/// absent in a given database edition or for any given IP.
#[derive(Debug, Clone, Default)]
pub struct GeoLookup {
    /// ISO 3166-1 alpha-2 code, e.g. "US"
    pub country: Option<String>,
    pub country_name: Option<String>,
    pub city: Option<String>,
    /// First-level subdivision (state / province)
    pub region: Option<String>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
}

impl GeoDb {
    /// Returns a no-op database that always returns None.
    pub fn empty() -> Self {
        Self { inner: None }
    }

    /// Load a MaxMind .mmdb file. Returns an error if the file can't be opened
    /// or parsed; the caller should handle this gracefully (e.g. via try_load).
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let reader = Reader::open_readfile(path)?;
        Ok(Self {
            inner: Some(Arc::new(reader)),
        })
    }

    /// Try to load a GeoIP database from an optional path.
    ///
    /// - `None` path → returns GeoDb::empty() silently.
    /// - Load error  → logs a warning and returns GeoDb::empty().
    ///
    /// Neither case is fatal.
    pub fn try_load(path: Option<&Path>) -> Self {
        match path {
            Some(p) => match Self::load(p) {
                Ok(db) => {
                    tracing::info!(?p, "GeoIP database loaded");
                    db
                }
                Err(e) => {
                    tracing::warn!(?p, error = %e, "GeoIP database failed to load — geo lookups will return None");
                    Self::empty()
                }
            },
            None => {
                tracing::info!("No GeoIP database path configured — geo lookups disabled");
                Self::empty()
            }
        }
    }

    /// Look up an IP address string. Returns None if:
    ///   • no database is loaded,
    ///   • the IP string can't be parsed,
    ///   • the IP isn't in the database.
    pub fn lookup(&self, ip_str: &str) -> Option<GeoLookup> {
        let reader = self.inner.as_ref()?;
        let ip: IpAddr = ip_str.parse().ok()?;

        // Use Country record for a lightweight lookup — always available in
        // both GeoLite2-City and GeoLite2-Country databases.
        // Try City first; fall back to Country-only if City decode fails.
        self.lookup_city(reader, ip)
            .or_else(|| self.lookup_country(reader, ip))
    }

    fn lookup_city(&self, reader: &Reader<Vec<u8>>, ip: IpAddr) -> Option<GeoLookup> {
        let record: maxminddb::geoip2::City<'_> = reader.lookup(ip).ok()?;

        let country = record
            .country
            .as_ref()
            .and_then(|c| c.iso_code)
            .map(|s| s.to_string());

        let country_name = record
            .country
            .as_ref()
            .and_then(|c| c.names.as_ref())
            .and_then(|n| n.get("en"))
            .map(|s| s.to_string());

        let city = record
            .city
            .as_ref()
            .and_then(|c| c.names.as_ref())
            .and_then(|n| n.get("en"))
            .map(|s| s.to_string());

        let region = record
            .subdivisions
            .as_ref()
            .and_then(|subs| subs.first())
            .and_then(|s| s.names.as_ref())
            .and_then(|n| n.get("en"))
            .map(|s| s.to_string());

        let latitude = record.location.as_ref().and_then(|l| l.latitude);
        let longitude = record.location.as_ref().and_then(|l| l.longitude);

        Some(GeoLookup {
            country,
            country_name,
            city,
            region,
            latitude,
            longitude,
        })
    }

    fn lookup_country(&self, reader: &Reader<Vec<u8>>, ip: IpAddr) -> Option<GeoLookup> {
        let record: maxminddb::geoip2::Country<'_> = reader.lookup(ip).ok()?;

        let country = record
            .country
            .as_ref()
            .and_then(|c| c.iso_code)
            .map(|s| s.to_string());

        let country_name = record
            .country
            .as_ref()
            .and_then(|c| c.names.as_ref())
            .and_then(|n| n.get("en"))
            .map(|s| s.to_string());

        Some(GeoLookup {
            country,
            country_name,
            ..Default::default()
        })
    }
}
