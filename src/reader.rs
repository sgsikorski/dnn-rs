use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::util::vector::VecN;

pub struct MnistDataset {
    pub train: Box<Vec<(VecN, VecN)>>,
    pub test: Box<Vec<(VecN, VecN)>>,
}

/// Load MNIST from two CSV files (train and test).
///
/// Expected CSV format per row: `label,pixel0,pixel1,...,pixel783`
/// An optional header row (where the first field is non-numeric) is skipped.
/// Pixel values are normalised to [0, 1]; labels are one-hot encoded over 10 classes.
pub fn load_mnist(train_path: &str, test_path: &str) -> Result<MnistDataset, String> {
    Ok(MnistDataset {
        train: load_csv(train_path)?,
        test: load_csv(test_path)?,
    })
}

/// Load a single MNIST CSV file and return (input, target) pairs.
pub fn load_csv(path: &str) -> Result<Box<Vec<(VecN, VecN)>>, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open '{}': {}", path, e))?;
    let reader = BufReader::new(file);
    let mut samples = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("Read error at line {}: {}", line_num + 1, e))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.splitn(2, ',').collect();

        // Skip header: first field is non-numeric (e.g. "label")
        if line_num == 0 && fields[0].parse::<u8>().is_err() {
            continue;
        }

        let (input, target) = parse_row(line, line_num + 1)?;
        samples.push((input, target));
    }

    if samples.is_empty() {
        return Err(format!("No samples found in '{}'", path));
    }

    Ok(Box::new(samples))
}

fn parse_row(line: &str, line_num: usize) -> Result<(VecN, VecN), String> {
    let mut fields = line.split(',');

    let label_str = fields
        .next()
        .ok_or_else(|| format!("Line {}: missing label", line_num))?;
    let label: usize = label_str
        .trim()
        .parse()
        .map_err(|_| format!("Line {}: invalid label '{}'", line_num, label_str))?;
    if label > 9 {
        return Err(format!(
            "Line {}: label {} out of range 0-9",
            line_num, label
        ));
    }

    let pixels: Vec<f64> = fields
        .enumerate()
        .map(|(i, f)| {
            f.trim()
                .parse::<f64>()
                .map(|v| v / 255.0)
                .map_err(|_| format!("Line {}: invalid pixel at column {}", line_num, i + 1))
        })
        .collect::<Result<Vec<_>, _>>()?;

    if pixels.len() != 784 {
        return Err(format!(
            "Line {}: expected 784 pixels, got {}",
            line_num,
            pixels.len()
        ));
    }

    let mut target = vec![0.0; 10];
    target[label] = 1.0;

    Ok((VecN::new(pixels), VecN::new(target)))
}
