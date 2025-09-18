use anyhow::{Context, Result, bail};
use oxigraph::io::RdfFormat;
use std::ffi::OsStr;
use std::path::Path;

pub fn format_from_path<T>(
    path: &Path,
    from_extension: impl FnOnce(&str) -> Result<T>,
) -> Result<T> {
    if let Some(ext) = path.extension().and_then(OsStr::to_str) {
        from_extension(ext).map_err(|e| {
            e.context(format!(
                "Not able to guess the file format from file name extension '{ext}'"
            ))
        })
    } else {
        bail!(
            "The path {} has no extension to guess a file format from",
            path.display()
        )
    }
}

pub fn rdf_format_from_path(path: &Path) -> Result<RdfFormat> {
    format_from_path(path, |ext| {
        RdfFormat::from_extension(ext)
            .with_context(|| format!("The file extension '{ext}' is unknown"))
    })
}

pub fn rdf_format_from_name(name: &str) -> Result<RdfFormat> {
    if let Some(t) = RdfFormat::from_extension(name) {
        return Ok(t);
    }
    if let Some(t) = RdfFormat::from_media_type(name) {
        return Ok(t);
    }
    bail!("The file format '{name}' is unknown")
}
