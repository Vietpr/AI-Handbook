---
title: "IoU → NMS: The Small Geometry Behind Object Detection"
description: "From overlapping boxes to the suppression rule that quietly shapes detector output."
domain: "Computer Vision"
topic: "Detection"
type: "Concept"
section: "Learn"
priority: "High"
order: 1
language: "en"
level: "Beginner"
pubDate: 2026-08-15
readingTime: 9
featured: false
draft: false
prerequisites:
  - Bounding boxes
---
Object detectors often predict several boxes around the same object. Non-Maximum Suppression turns that noisy set into a smaller set of useful detections.

## Intersection over Union

IoU measures how much two boxes overlap relative to their combined area.

A value near zero means little overlap. A value near one means the boxes are nearly identical.

## Why detectors produce duplicates

Dense prediction creates many candidate boxes. Several neighboring predictions can all assign high confidence to the same physical object.

That is expected behavior before post-processing.

## NMS as a greedy rule

A classic NMS loop is intentionally simple:

1. Keep the highest-confidence box.
2. Compare it with the remaining boxes.
3. Suppress boxes whose IoU crosses a threshold.
4. Repeat with the next surviving box.

```python
while boxes:
    best = pop_highest_score(boxes)
    keep.append(best)
    boxes = [b for b in boxes if iou(best, b) < threshold]
```

## The threshold is a product decision

A low threshold suppresses aggressively and can merge nearby objects. A high threshold preserves more candidates and can leave duplicates.

So NMS is not merely a mathematical cleanup step. It encodes what your application considers "the same detection."
