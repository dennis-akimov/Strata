//! A My Maps map kept on this Mac: its KML (what My Maps exports: each place's name, its columns as ExtendedData and
//! the point Google placed it at), a GeoJSON copy (RFC 7946: [longitude, latitude]), and the coordinates written into
//! the table it came from, as Lat / Lng by its key (Link), so the table knows where each place is.
use serde_json::{json, Value};

/// One place of the map.
#[derive(Debug, Clone, PartialEq)]
pub struct Place {
    pub name: String,
    pub lng: f64,
    pub lat: f64,
    pub data: Vec<(String, String)>,
}

/// XML text: the five entities and CDATA (what My Maps writes).
fn text(s: &str) -> String {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix("<![CDATA[").and_then(|r| r.strip_suffix("]]>")) {
        return inner.trim().to_string();
    }
    s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'").replace("&amp;", "&")
}

/// The text of the first <tag>...</tag> in `block` (tags with attributes too).
fn inner<'a>(block: &'a str, tag: &str) -> Option<&'a str> {
    let open = block.find(&format!("<{tag}>")).map(|i| i + tag.len() + 2)
        .or_else(|| block.find(&format!("<{tag} ")).and_then(|i| block[i..].find('>').map(|j| i + j + 1)))?;
    let close = block[open..].find(&format!("</{tag}>"))?;
    Some(&block[open..open + close])
}

/// The places of a KML document: every Placemark with a Point.
pub fn parse_kml(kml: &str) -> Vec<Place> {
    let mut out = vec![];
    let mut rest = kml;
    while let Some(i) = rest.find("<Placemark") {
        let Some(j) = rest[i..].find("</Placemark>") else { break };
        let block = &rest[i..i + j];
        rest = &rest[i + j + 12..];
        let Some(coords) = inner(block, "coordinates") else { continue };   // a line or a shape: not a place
        let nums: Vec<f64> = coords.split(',').filter_map(|n| n.trim().parse().ok()).collect();
        if nums.len() < 2 {
            continue;
        }
        let mut data = vec![];
        let mut d = block;
        while let Some(k) = d.find("<Data name=\"") {
            let after = &d[k + 12..];
            let Some(q) = after.find('"') else { break };
            let name = text(&after[..q]);
            let end = after.find("</Data>").unwrap_or(after.len());
            data.push((name, inner(&after[..end], "value").map(text).unwrap_or_default()));
            d = &after[end..];
        }
        out.push(Place { name: inner(block, "name").map(text).unwrap_or_default(), lng: nums[0], lat: nums[1], data });
    }
    out
}

/// The places as an RFC 7946 FeatureCollection (coordinates longitude first).
pub fn geojson(places: &[Place]) -> Value {
    json!({"type": "FeatureCollection", "features": places.iter().map(|p| {
        let mut props = serde_json::Map::new();
        props.insert("name".into(), json!(p.name));
        for (k, v) in &p.data {
            props.insert(k.clone(), json!(v));
        }
        json!({"type": "Feature", "geometry": {"type": "Point", "coordinates": [p.lng, p.lat]}, "properties": props})
    }).collect::<Vec<_>>()})
}

/// Lat / Lng into the table's rows whose key (Link) a place carries; the columns are added when missing.
/// (rows given coordinates, places that matched no row)
pub fn fill_lat_lng(table: &mut crate::table::Table, places: &[Place], key: &str) -> Result<(usize, usize), String> {
    let header = table.first_mut().ok_or("the table is empty")?;
    let col = |h: &[String], n: &str| h.iter().position(|x| x.trim().eq_ignore_ascii_case(n));
    let k = col(header, key).ok_or_else(|| format!("the table has no {key:?} column"))?;
    for c in ["Lat", "Lng"] {
        if col(header, c).is_none() {
            header.push(c.to_string());
        }
    }
    let (lat, lng, width) = (col(header, "Lat").unwrap(), col(header, "Lng").unwrap(), header.len());
    let (mut filled, mut unmatched) = (0, 0);
    for p in places {
        let Some(kv) = p.data.iter().find(|(n, _)| n.trim().eq_ignore_ascii_case(key)).map(|(_, v)| v.trim()).filter(|v| !v.is_empty()) else {
            unmatched += 1;
            continue;
        };
        match table.iter_mut().skip(1).find(|r| r.get(k).is_some_and(|x| x.trim() == kv)) {
            Some(row) => {
                row.resize(width, String::new());
                row[lat] = format!("{:.6}", p.lat);
                row[lng] = format!("{:.6}", p.lng);
                filled += 1;
            }
            None => unmatched += 1,
        }
    }
    Ok((filled, unmatched))
}

#[cfg(test)]
mod tests {
    use super::*;

    const KML: &str = r#"<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>studios</name>
<Folder><name>studios</name>
<Placemark><name>Luma Apartments: Little Italy high-rise</name><description><![CDATA[Rating: 4.0]]></description>
<ExtendedData><Data name="Rating"><value>4.0 (122)</value></Data><Data name="Link"><value>https://www.rentluma.com/floorplans</value></Data>
<Data name="Price"><value>$2,916&amp;more</value></Data></ExtendedData>
<Point><coordinates>
-117.1675,32.7215,0
</coordinates></Point></Placemark>
<Placemark><name>A line</name><LineString><tessellate>1</tessellate></LineString></Placemark>
<Placemark><name><![CDATA[City Villas]]></name><ExtendedData><Data name="Link"><value>https://www.cityvillassandiego.com/floorplans</value></Data></ExtendedData>
<Point><coordinates>-117.1490,32.7150,0</coordinates></Point></Placemark>
<Placemark><name>Not in the table</name><ExtendedData><Data name="Link"><value>https://elsewhere</value></Data></ExtendedData><Point><coordinates>-117,32,0</coordinates></Point></Placemark>
</Folder></Document></kml>"#;

    #[test]
    fn places_from_my_maps_kml() {
        let p = parse_kml(KML);
        assert_eq!(p.len(), 3, "a line is not a place");
        assert_eq!(p[0].name, "Luma Apartments: Little Italy high-rise");
        assert_eq!((p[0].lng, p[0].lat), (-117.1675, 32.7215));
        assert_eq!(p[0].data, [("Rating".to_string(), "4.0 (122)".to_string()), ("Link".into(), "https://www.rentluma.com/floorplans".into()), ("Price".into(), "$2,916&more".into())]);
        assert_eq!(p[1].name, "City Villas", "CDATA names");
        // GeoJSON: longitude first (RFC 7946 A.1)
        let g = geojson(&p);
        assert_eq!(g["features"][0]["geometry"], json!({"type": "Point", "coordinates": [-117.1675, 32.7215]}));
        assert_eq!(g["features"][0]["properties"]["Link"], "https://www.rentluma.com/floorplans");
    }

    #[test]
    fn coordinates_into_the_table_by_link() {
        let mut t: crate::table::Table = vec![
            vec!["Address".into(), "Link".into()],
            vec!["1440 Columbia St".into(), "https://www.rentluma.com/floorplans".into()],
            vec!["845 16th St".into(), "https://www.cityvillassandiego.com/floorplans".into()],
            vec!["no place yet".into(), "https://none".into()],
        ];
        assert_eq!(fill_lat_lng(&mut t, &parse_kml(KML), "Link").unwrap(), (2, 1));
        assert_eq!(t[0], ["Address", "Link", "Lat", "Lng"]);
        assert_eq!(t[1], ["1440 Columbia St", "https://www.rentluma.com/floorplans", "32.721500", "-117.167500"]);
        assert_eq!(t[3].len(), 2, "a row with no place keeps its cells (empty Lat/Lng when written)");
        // again: the columns are not added twice
        assert_eq!(fill_lat_lng(&mut t, &parse_kml(KML), "Link").unwrap(), (2, 1));
        assert_eq!(t[0].len(), 4);
    }
}
