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
        let record: maxminddb::geoip2::City<'_> = reader.lookup(ip).ok()?.decode().ok()??;

        Some(GeoLookup {
            country: record.country.iso_code.map(str::to_string),
            country_name: record.country.names.english.map(str::to_string),
            city: record.city.names.english.map(str::to_string),
            region: record
                .subdivisions
                .first()
                .and_then(|s| s.names.english)
                .map(str::to_string),
            latitude: record.location.latitude,
            longitude: record.location.longitude,
        })
    }

    fn lookup_country(&self, reader: &Reader<Vec<u8>>, ip: IpAddr) -> Option<GeoLookup> {
        let record: maxminddb::geoip2::Country<'_> = reader.lookup(ip).ok()?.decode().ok()??;

        Some(GeoLookup {
            country: record.country.iso_code.map(str::to_string),
            country_name: record.country.names.english.map(str::to_string),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_db_returns_none() {
        let db = GeoDb::empty();
        assert!(db.lookup("89.160.20.112").is_none());
        assert!(db.lookup("not an ip").is_none());
        assert!(GeoDb::try_load(None).lookup("89.160.20.112").is_none());
    }

    #[test]
    fn unreadable_db_is_an_error_and_try_load_falls_back() {
        let dir = std::env::temp_dir().join(format!("incord-geoip-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let garbage = dir.join("garbage.mmdb");
        std::fs::write(&garbage, b"not a maxmind database").unwrap();
        assert!(GeoDb::load(&garbage).is_err());
        assert!(GeoDb::load(&dir.join("missing.mmdb")).is_err());
        assert!(GeoDb::try_load(Some(&garbage)).lookup("89.160.20.112").is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Decodes a real database. Needs MaxMind's public test databases
    /// (github.com/maxmind/MaxMind-DB, test-data/): set
    /// INCORD_GEOIP_TEST_DIR to the folder holding GeoIP2-City-Test.mmdb and
    /// GeoIP2-Country-Test.mmdb, then run with `--ignored`.
    #[test]
    #[ignore]
    fn decodes_maxmind_test_databases() {
        let dir = std::path::PathBuf::from(
            std::env::var("INCORD_GEOIP_TEST_DIR").expect("set INCORD_GEOIP_TEST_DIR"),
        );
        let city = GeoDb::load(&dir.join("GeoIP2-City-Test.mmdb")).unwrap();
        let hit = city.lookup("89.160.20.112").expect("IP is in the City test database");
        assert_eq!(hit.country.as_deref(), Some("SE"));
        assert_eq!(hit.country_name.as_deref(), Some("Sweden"));
        assert!(hit.latitude.is_some() && hit.longitude.is_some());
        assert!(city.lookup("10.0.0.1").is_none(), "private IP is not in the database");

        let country = GeoDb::load(&dir.join("GeoIP2-Country-Test.mmdb")).unwrap();
        let hit = country.lookup("89.160.20.112").expect("IP is in the Country test database");
        assert_eq!(hit.country.as_deref(), Some("SE"));
        assert!(hit.city.is_none());
    }
}
