//! Structural decode: the shapes, refinement parents and placements of a JB2 image,
//! instead of its composited bitmap.
//!
//! Re-encoding a JB2 layer in another symbol-based format (e.g. JBIG2) losslessly needs
//! exactly this: which bitmaps exist, which one each refinement was predicted from, and
//! where every bitmap is drawn. [`decode_structure`] runs the same record loop as
//! [`decode`](crate::decode) — the symbol library evolves identically — and records
//! shapes and blits rather than painting them. Compositing the blits (OR) reproduces
//! `decode()`'s bitmap exactly.

use super::*;

/// One shape: a tight bitmap (no empty border rows/columns) and the shape it was
/// refined from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Jb2Shape {
    /// Pixels, top row first (same layout as [`decode`](crate::decode)'s output).
    pub bitmap: Bitmap,
    /// Index of the shape this one was coded as a refinement of (JB2 record types
    /// 4/5/6), or `None` if it was coded directly. Always smaller than this shape's own
    /// index, in the same index space as [`Jb2Blit::shape`].
    pub parent: Option<usize>,
}

/// One placement of a shape on the page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Jb2Blit {
    /// Shape index: `< inherited` refers to the shared dictionary
    /// ([`Jb2Dict::shape`]), otherwise to `shapes[shape - inherited]`.
    pub shape: usize,
    /// Column of the shape's left edge.
    pub left: i32,
    /// Row of the shape's bottom edge, counted from the bottom of the page (DjVu
    /// convention, as in djvulibre's `JB2Blit`). May lie outside the page.
    pub bottom: i32,
}

/// The structure of one JB2 image (Sjbz stream).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Jb2Structure {
    /// Page width in pixels.
    pub width: u32,
    /// Page height in pixels.
    pub height: u32,
    /// Number of shapes taken from the shared dictionary; shape indices
    /// `0..inherited` refer to it.
    pub inherited: usize,
    /// Shapes defined by this stream (index `inherited + i`).
    pub shapes: Vec<Jb2Shape>,
    /// Placements in stream order; compositing them with OR gives the page.
    pub blits: Vec<Jb2Blit>,
}

impl Jb2Structure {
    /// Composite the blits (OR) into a page bitmap. Equal to [`decode`](crate::decode).
    pub fn render(&self, dict: Option<&Jb2Dict>) -> Bitmap {
        let mut page = Bitmap::new(self.width, self.height);
        for b in &self.blits {
            let bm = if b.shape < self.inherited {
                match dict {
                    Some(d) => d.shape(b.shape).bitmap,
                    None => continue,
                }
            } else {
                self.shapes[b.shape - self.inherited].bitmap.clone()
            };
            for y in 0..bm.height {
                // shape row y (top-down) sits on page row (bottom-up) b.bottom + h-1-y
                let page_row_up = b.bottom + (bm.height - 1 - y) as i32;
                if page_row_up < 0 || page_row_up >= self.height as i32 {
                    continue;
                }
                let py = self.height - 1 - page_row_up as u32;
                for x in 0..bm.width {
                    let px = b.left + x as i32;
                    if px >= 0 && (px as u32) < self.width && bm.get(x, y) {
                        page.set_black(px as u32, py);
                    }
                }
            }
        }
        page
    }
}

impl Jb2Dict {
    /// Number of shapes in the dictionary (including any it inherited).
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Whether the dictionary has no shapes.
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// Shape `index` (bitmap + refinement parent).
    ///
    /// # Panics
    /// If `index >= self.len()`.
    pub fn shape(&self, index: usize) -> Jb2Shape {
        Jb2Shape {
            bitmap: jbm_to_bitmap(&self.symbols[index]),
            parent: self.parents[index].map(|p| p as usize),
        }
    }
}

/// `Jbm` (bottom row first) → `Bitmap` (top row first).
fn jbm_to_bitmap(j: &Jbm) -> Bitmap {
    let (w, h) = (j.width.max(0) as u32, j.height.max(0) as u32);
    let mut out = Bitmap::new(w, h);
    let src_stride = j.stride();
    let dst_stride = out.row_stride();
    for row in 0..h as usize {
        let src = &j.data[row * src_stride..(row + 1) * src_stride];
        let y = h as usize - 1 - row;
        out.data[y * dst_stride..y * dst_stride + src_stride].copy_from_slice(src);
    }
    out
}

/// Bounding box of the black pixels: (min_col, min_row) in `Jbm` coordinates, or
/// `None` for an all-white bitmap.
fn content_origin(j: &Jbm) -> Option<(i32, i32)> {
    let mut min_row = None;
    let mut min_col = i32::MAX;
    for row in 0..j.height {
        for col in 0..j.width {
            if j.get(row, col) != 0 {
                min_row.get_or_insert(row);
                min_col = min_col.min(col);
                break; // first black pixel of this row is its leftmost
            }
        }
    }
    min_row.map(|r| (min_col, r))
}

/// Decode a JB2 image stream into its shapes and placements.
///
/// `shared_dict` must be provided when the stream inherits from a shared dictionary,
/// exactly as for [`decode`](crate::decode).
pub fn decode_structure(
    data: &[u8],
    shared_dict: Option<&Jb2Dict>,
) -> Result<Jb2Structure, Jb2Error> {
    let mut pool: Vec<u8> = Vec::new();
    let pool = &mut pool;
    let mut zp = ZpDecoder::new(data).map_err(|_| Jb2Error::ZpInitFailed)?;

    let mut record_type_ctx = NumContext::new();
    let mut image_size_ctx = NumContext::new();
    let mut symbol_width_ctx = NumContext::new();
    let mut symbol_height_ctx = NumContext::new();
    let mut inherit_dict_size_ctx = NumContext::new();
    let mut coord_ctx = CoordContexts::new();
    let mut symbol_index_ctx = NumContext::new();
    let mut symbol_width_diff_ctx = NumContext::new();
    let mut symbol_height_diff_ctx = NumContext::new();
    let mut horiz_abs_loc_ctx = NumContext::new();
    let mut vert_abs_loc_ctx = NumContext::new();
    let mut comment_length_ctx = NumContext::new();
    let mut comment_octet_ctx = NumContext::new();

    let mut direct_bitmap_ctx = [0u8; 1024];
    let mut refinement_bitmap_ctx = [0u8; 2048];
    let mut refinement_bitmap_ctx_p = [0x8000u16; 2048];
    let mut total_sym_pixels = 0usize;
    let mut total_blit_pixels = 0usize;

    let mut rtype = decode_num(&mut zp, &mut record_type_ctx, 0, 11);
    let mut initial_dict_length: usize = 0;
    if rtype == 9 {
        initial_dict_length = decode_num(&mut zp, &mut inherit_dict_size_ctx, 0, 262142) as usize;
        rtype = decode_num(&mut zp, &mut record_type_ctx, 0, 11);
    }
    let _ = rtype;

    let image_width = {
        let w = decode_num(&mut zp, &mut image_size_ctx, 0, 262142);
        if w == 0 { 200 } else { w }
    };
    let image_height = {
        let h = decode_num(&mut zp, &mut image_size_ctx, 0, 262142);
        if h == 0 { 200 } else { h }
    };
    let mut flag_ctx: u8 = 0;
    if zp.decode_bit(&mut flag_ctx) {
        return Err(Jb2Error::BadHeaderFlag);
    }

    let initial_symbols: &[Jbm] = if initial_dict_length > 0 {
        match shared_dict {
            Some(sd) => {
                if initial_dict_length > sd.symbols.len() {
                    return Err(Jb2Error::InheritedDictTooLarge);
                }
                &sd.symbols[..initial_dict_length]
            }
            None => return Err(Jb2Error::MissingSharedDict),
        }
    } else {
        &[]
    };
    let mut dict = JbmDict::new(initial_symbols);
    const MAX_PIXELS: usize = 64 * 1024 * 1024;
    if (image_width as usize).saturating_mul(image_height as usize) > MAX_PIXELS {
        return Err(Jb2Error::ImageTooLarge);
    }

    // library index (as used by the stream) -> shape index
    let mut lib: Vec<usize> = (0..initial_dict_length).collect();
    let mut shapes: Vec<Jb2Shape> = Vec::new();
    let mut blits: Vec<Jb2Blit> = Vec::new();
    let inherited = initial_dict_length;
    let mut layout = LayoutState::new(image_height);

    // Store `bm` cropped to its content as a new shape. Returns the shape index and the
    // crop origin (column, row from the bottom), or `None` for an all-white bitmap.
    let add_shape = |shapes: &mut Vec<Jb2Shape>,
                     bm: &Jbm,
                     parent: Option<usize>|
     -> Option<(usize, i32, i32)> {
        let (dx, dy) = content_origin(bm)?;
        shapes.push(Jb2Shape {
            bitmap: jbm_to_bitmap(&bm.crop_to_content()),
            parent,
        });
        Some((inherited + shapes.len() - 1, dx, dy))
    };

    let max_sym_px = MAX_PAGE_SYMBOL_WORK;
    let mut record_count = 0usize;
    loop {
        if zp.synthetic_bytes() > ZP_EOF_SLACK_BYTES {
            return Err(Jb2Error::Truncated);
        }
        if record_count >= MAX_RECORDS {
            return Err(Jb2Error::TooManyRecords);
        }
        record_count += 1;
        let rtype = decode_num(&mut zp, &mut record_type_ctx, 0, 11);
        match rtype {
            // 1 / 2 / 3 — new symbol, direct coding: 1 dict + blit, 2 dict only, 3 blit only
            1..=3 => {
                let w = decode_num(&mut zp, &mut symbol_width_ctx, 0, 262142);
                let h = decode_num(&mut zp, &mut symbol_height_ctx, 0, 262142);
                check_symbol_decode_budget(&zp, w, h, 1, &mut total_sym_pixels, max_sym_px)?;
                let bm = decode_bitmap_direct(&mut zp, &mut direct_bitmap_ctx, w, h, pool)?;
                let shape = add_shape(&mut shapes, &bm, None);
                if rtype != 2 {
                    let (x, y) = decode_symbol_coords(
                        &mut zp,
                        &mut coord_ctx,
                        &mut layout,
                        bm.width,
                        bm.height,
                    );
                    check_blit_budget(&bm, &mut total_blit_pixels)?;
                    if let Some((s, dx, dy)) = shape {
                        blits.push(Jb2Blit {
                            shape: s,
                            left: x + dx,
                            bottom: y + dy,
                        });
                    }
                }
                if rtype == 3 {
                    bm.recycle_into(pool);
                } else {
                    // the library keeps the cropped bitmap: exactly what decode() stores
                    lib.push(shape.map_or(usize::MAX, |(s, _, _)| s));
                    dict.push(bm.crop_and_recycle(pool));
                }
            }

            // 4 / 5 / 6 — refinement of a library symbol: 4 dict + blit, 5 dict only, 6 blit only
            4..=6 => {
                if dict.is_empty() {
                    return Err(Jb2Error::EmptyDictReference);
                }
                let index =
                    decode_num(&mut zp, &mut symbol_index_ctx, 0, dict.len() as i32 - 1) as usize;
                if index >= dict.len() {
                    return Err(Jb2Error::InvalidSymbolIndex);
                }
                let wdiff = decode_num(&mut zp, &mut symbol_width_diff_ctx, -262143, 262142);
                let hdiff = decode_num(&mut zp, &mut symbol_height_diff_ctx, -262143, 262142);
                let cbm_w = dict[index].width + wdiff;
                let cbm_h = dict[index].height + hdiff;
                check_symbol_decode_budget(
                    &zp,
                    cbm_w,
                    cbm_h,
                    REFINE_PIXEL_WORK,
                    &mut total_sym_pixels,
                    max_sym_px,
                )?;
                let cbm = decode_bitmap_ref(
                    &mut zp,
                    &mut refinement_bitmap_ctx,
                    &mut refinement_bitmap_ctx_p,
                    cbm_w,
                    cbm_h,
                    &dict[index],
                    pool,
                )?;
                // an all-white parent has no shape: code the child directly
                let parent = Some(lib[index]).filter(|&p| p != usize::MAX);
                let shape = add_shape(&mut shapes, &cbm, parent);
                if rtype != 5 {
                    let (x, y) = decode_symbol_coords(
                        &mut zp,
                        &mut coord_ctx,
                        &mut layout,
                        cbm.width,
                        cbm.height,
                    );
                    check_blit_budget(&cbm, &mut total_blit_pixels)?;
                    if let Some((s, dx, dy)) = shape {
                        blits.push(Jb2Blit {
                            shape: s,
                            left: x + dx,
                            bottom: y + dy,
                        });
                    }
                }
                if rtype == 6 {
                    cbm.recycle_into(pool);
                } else {
                    lib.push(shape.map_or(usize::MAX, |(s, _, _)| s));
                    dict.push(cbm.crop_and_recycle(pool));
                }
            }

            // 7 — matched copy of a library symbol (already cropped), blit only
            7 => {
                if dict.is_empty() {
                    return Err(Jb2Error::EmptyDictReference);
                }
                let index =
                    decode_num(&mut zp, &mut symbol_index_ctx, 0, dict.len() as i32 - 1) as usize;
                if index >= dict.len() {
                    return Err(Jb2Error::InvalidSymbolIndex);
                }
                let (bm_w, bm_h) = (dict[index].width, dict[index].height);
                let (x, y) = decode_symbol_coords(&mut zp, &mut coord_ctx, &mut layout, bm_w, bm_h);
                check_blit_budget(&dict[index], &mut total_blit_pixels)?;
                if lib[index] != usize::MAX && bm_w > 0 && bm_h > 0 {
                    blits.push(Jb2Blit {
                        shape: lib[index],
                        left: x,
                        bottom: y,
                    });
                }
            }

            // 8 — non-symbol data at an absolute position, blit only
            8 => {
                let w = decode_num(&mut zp, &mut symbol_width_ctx, 0, 262142);
                let h = decode_num(&mut zp, &mut symbol_height_ctx, 0, 262142);
                check_symbol_decode_budget(&zp, w, h, 1, &mut total_sym_pixels, max_sym_px)?;
                let bm = decode_bitmap_direct(&mut zp, &mut direct_bitmap_ctx, w, h, pool)?;
                let left = decode_num(&mut zp, &mut horiz_abs_loc_ctx, 1, image_width);
                let top = decode_num(&mut zp, &mut vert_abs_loc_ctx, 1, image_height);
                check_blit_budget(&bm, &mut total_blit_pixels)?;
                if let Some((s, dx, dy)) = add_shape(&mut shapes, &bm, None) {
                    blits.push(Jb2Blit {
                        shape: s,
                        left: left - 1 + dx,
                        bottom: top - h + dy,
                    });
                }
                bm.recycle_into(pool);
            }

            9 => {}

            // 10 — comment: every octet must be consumed to keep the ZP decoder in sync
            10 => {
                let length = decode_num(&mut zp, &mut comment_length_ctx, 0, 262142) as usize;
                for _ in 0..length {
                    decode_num(&mut zp, &mut comment_octet_ctx, 0, 255);
                }
            }

            11 => break,

            _ => return Err(Jb2Error::UnknownRecordType),
        }
    }

    Ok(Jb2Structure {
        width: image_width as u32,
        height: image_height as u32,
        inherited,
        shapes,
        blits,
    })
}
