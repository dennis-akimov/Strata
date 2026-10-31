//! A My Maps map kept on this Mac: its KML (what My Maps exports: each place's name, address and columns as
//! ExtendedData), a GeoJSON copy (RFC 7946: [longitude, latitude]), and the coordinates written into the table it came
//! from, as Lat / Lng by its key (Link).  My Maps' KML has no coordinates for places it placed by address (tried live:
//! 21 placemarks, 0 <coordinates>); the map's page has them (its _pageData, "geocoded"): they are added by Link.
use serde_json::{json, Value};

/// One place of the map.
#[derive(Debug, Clone, PartialEq)]
pub struct Place {
    pub name: String,
    pub lng: Option<f64>,
    pub lat: Option<f64>,
    pub data: Vec<(String, String)>,
}

impl Place {
    pub fn get(&self, column: &str) -> Option<&str> {
        self.data.iter().find(|(n, _)| n.trim().eq_ignore_ascii_case(column)).map(|(_, v)| v.trim()).filter(|v| !v.is_empty())
    }
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

/// The places of a KML document: every Placemark with a point or an address (a line or a shape is no place).
pub fn parse_kml(kml: &str) -> Vec<Place> {
    let mut out = vec![];
    let mut rest = kml;
    while let Some(i) = rest.find("<Placemark") {
        let Some(j) = rest[i..].find("</Placemark>") else { break };
        let block = &rest[i..i + j];
        rest = &rest[i + j + 12..];
        let nums: Vec<f64> = inner(block, "coordinates").map(|c| c.split(',').filter_map(|n| n.trim().parse().ok()).collect()).unwrap_or_default();
        if nums.len() < 2 && inner(block, "address").is_none() {
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
        let (lng, lat) = if nums.len() >= 2 { (Some(nums[0]), Some(nums[1])) } else { (None, None) };
        out.push(Place { name: inner(block, "name").map(text).unwrap_or_default(), lng, lat, data });
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
        // a place with no point yet: geometry null, as RFC 7946 3.2 allows
        let geometry = match (p.lng, p.lat) { (Some(x), Some(y)) => json!({"type": "Point", "coordinates": [x, y]}), _ => Value::Null };
        json!({"type": "Feature", "geometry": geometry, "properties": props})
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
        let (Some(kv), Some(y), Some(x)) = (p.get(key), p.lat, p.lng) else {
            unmatched += 1;
            continue;
        };
        match table.iter_mut().skip(1).find(|r| r.get(k).is_some_and(|x| x.trim() == kv)) {
            Some(row) => {
                row.resize(width, String::new());
                row[lat] = format!("{y:.6}");
                row[lng] = format!("{x:.6}");
                filled += 1;
            }
            None => unmatched += 1,
        }
    }
    Ok((filled, unmatched))
}

/// The points the map's page knows (by key) into places that have none.  How many got one.
pub fn add_points(places: &mut [Place], page: &[Place], key: &str) -> usize {
    let mut n = 0;
    for p in places.iter_mut().filter(|p| p.lat.is_none()) {
        if let Some(q) = p.get(key).and_then(|k| page.iter().find(|q| q.get(key) == Some(k) && q.lat.is_some())) {
            (p.lat, p.lng) = (q.lat, q.lng);
            n += 1;
        }
    }
    n
}

/// The KML with a <Point> for each placemark that had none, from `places` by key (a KML for any map app).
pub fn kml_with_points(kml: &str, places: &[Place], key: &str) -> String {
    let mut out = String::with_capacity(kml.len() + places.len() * 80);
    let mut rest = kml;
    while let Some(i) = rest.find("</Placemark>") {
        let block = &rest[..i];
        let start = block.rfind("<Placemark").unwrap_or(0);
        let mark = &block[start..];
        out.push_str(block);
        if !mark.contains("<coordinates>") {
            let k = mark.find(&format!("<Data name=\"{key}\">")).and_then(|d| inner(&mark[d..], "value")).map(text);
            if let Some(p) = k.and_then(|k| places.iter().find(|p| p.get(key) == Some(k.trim()))) {
                if let (Some(x), Some(y)) = (p.lng, p.lat) {
                    out.push_str(&format!("<Point><coordinates>{x},{y},0</coordinates></Point>"));
                }
            }
        }
        out.push_str("</Placemark>");
        rest = &rest[i + 12..];
    }
    out.push_str(rest);
    out
}

/// The map page's places (its script's {lat, lng, data}) as Places.
pub fn page_places(v: &Value) -> Vec<Place> {
    v.as_array().into_iter().flatten().filter_map(|p| Some(Place {
        name: String::new(),
        lat: Some(p["lat"].as_f64()?),
        lng: Some(p["lng"].as_f64()?),
        data: p["data"].as_object()?.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect(),
    })).collect()
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
        assert_eq!((p[0].lng, p[0].lat), (Some(-117.1675), Some(32.7215)));
        assert_eq!(p[0].data, [("Rating".to_string(), "4.0 (122)".to_string()), ("Link".into(), "https://www.rentluma.com/floorplans".into()), ("Price".into(), "$2,916&more".into())]);
        assert_eq!(p[1].name, "City Villas", "CDATA names");
        // GeoJSON: longitude first (RFC 7946 A.1)
        let g = geojson(&p);
        assert_eq!(g["features"][0]["geometry"], json!({"type": "Point", "coordinates": [-117.1675, 32.7215]}));
        assert_eq!(g["features"][0]["properties"]["Link"], "https://www.rentluma.com/floorplans");
    }

    // My Maps' KML for places placed by address: <address>, the columns, no <coordinates> (as exported live)
    const KML_NO_POINTS: &str = r#"<kml><Document><Folder><Placemark><name>Luma</name><address>1440 Columbia St, San Diego, CA 92101</address>
<ExtendedData><Data name="Link"><value>https://www.rentluma.com/floorplans</value></Data><Data name="Time"><value/></Data></ExtendedData></Placemark>
<Placemark><name>Gone</name><address>1 Nowhere</address><ExtendedData><Data name="Link"><value>https://gone</value></Data></ExtendedData></Placemark></Folder></Document></kml>"#;

    #[test]
    fn points_from_the_page_into_a_kml_without_them() {
        let mut p = parse_kml(KML_NO_POINTS);
        assert_eq!(p.len(), 2, "placemarks with an address are places");
        assert_eq!((p[0].lat, p[0].get("Time")), (None, None), "no point yet; an empty <value/> is no value");
        let page = page_places(&json!([{"lat": 32.7203183, "lng": -117.1676694, "data": {"Link": "https://www.rentluma.com/floorplans"}}]));
        assert_eq!(add_points(&mut p, &page, "Link"), 1);
        assert_eq!((p[0].lat, p[0].lng), (Some(32.7203183), Some(-117.1676694)));
        let k = kml_with_points(KML_NO_POINTS, &p, "Link");
        assert!(k.contains("<Point><coordinates>-117.1676694,32.7203183,0</coordinates></Point></Placemark>"), "{k}");
        assert_eq!(k.matches("<Point>").count(), 1, "no point for a place the page does not know");
        assert_eq!(parse_kml(&k)[0].lat, Some(32.7203183), "the written KML reads back with its point");
        assert_eq!(geojson(&p)["features"][1]["geometry"], Value::Null, "no point: geometry null");
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
