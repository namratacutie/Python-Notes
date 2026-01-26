import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create figure and axis with black background
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.patch.set_facecolor('#0a0a0a')
ax.set_facecolor('#0a0a0a')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')

# ============================================
# 1. BACKGROUND CIRCLE (dusty mauve/pink)
# ============================================
bg_circle = patches.Circle((5, 5), 4.5, linewidth=0, facecolor='#c4a0a0')
ax.add_patch(bg_circle)

# Clipping circle for all elements
clip_circle = patches.Circle((5, 5), 4.5, transform=ax.transData)

# ============================================
# 2. HAIR - Dark brown/black messy hair
# ============================================
# Main hair mass - covers top and sides
hair_main = patches.Ellipse((5, 6), 9, 6, linewidth=0, facecolor='#2a2320', zorder=2)
hair_main.set_clip_path(clip_circle)
ax.add_patch(hair_main)


left_hair_x = [0.5, 0.3, 0.5, 0.8, 1.2, 1.8, 2.2, 2.5, 2.8, 2.5, 2.2, 1.8, 1.2, 0.8, 0.5]
left_hair_y = [7.5, 5.5, 3.5, 2.5, 1.8, 1.5, 1.8, 2.5, 4.0, 5.5, 6.5, 7.5, 8.0, 8.0, 7.5]
hair_left_patch = plt.Polygon(list(zip(left_hair_x, left_hair_y)), facecolor='#2a2320', zorder=2)
hair_left_patch.set_clip_path(clip_circle)
ax.add_patch(hair_left_patch)


right_hair_x = [9.5, 9.7, 9.5, 9.2, 8.8, 8.2, 7.8, 7.5, 7.2, 7.5, 7.8, 8.2, 8.8, 9.2, 9.5]
right_hair_y = [7.5, 5.5, 3.5, 2.5, 1.8, 1.5, 1.8, 2.5, 4.0, 5.5, 6.5, 7.5, 8.0, 8.0, 7.5]
hair_right_patch = plt.Polygon(list(zip(right_hair_x, right_hair_y)), facecolor='#2a2320', zorder=2)
hair_right_patch.set_clip_path(clip_circle)
ax.add_patch(hair_right_patch)


face = patches.Ellipse((5, 4.5), 5.2, 5.8, linewidth=0, facecolor='#f5e2c8', zorder=3)
face.set_clip_path(clip_circle)
ax.add_patch(face)

# ============================================
# 4. HAIR BANGS - Wavy forehead coverage
# ============================================
bangs_x = [2.0, 2.3, 2.6, 3.0, 3.3, 3.6, 4.0, 4.3, 4.6, 5.0, 5.3, 5.6, 6.0, 6.3, 6.6, 7.0, 7.3, 7.6, 8.0,
           8.0, 7.6, 7.3, 7.0, 6.6, 6.3, 6.0, 5.6, 5.3, 5.0, 4.6, 4.3, 4.0, 3.6, 3.3, 3.0, 2.6, 2.3, 2.0]
bangs_y = [5.8, 6.1, 5.9, 6.2, 6.0, 6.3, 6.1, 6.4, 6.2, 6.5, 6.3, 6.1, 6.4, 6.2, 6.0, 5.8, 5.6, 5.5, 5.4,
           10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
bangs_patch = plt.Polygon(list(zip(bangs_x, bangs_y)), facecolor='#2a2320', zorder=4)
bangs_patch.set_clip_path(clip_circle)
ax.add_patch(bangs_patch)

# ============================================
# 5. PINK BOW/CLIP - Upper left with dots
# ============================================
bow_x = [1.6, 2.0, 2.4, 2.8, 3.1, 2.8, 2.4, 2.0, 1.6, 1.4, 1.6]
bow_y = [7.0, 7.4, 7.6, 7.3, 6.9, 6.6, 6.8, 6.6, 6.8, 6.9, 7.0]
ax.fill(bow_x, bow_y, color='#c090a8', zorder=5)

# Bow dots/pattern (darker spots)
bow_dots = [(1.9, 7.1), (2.2, 7.3), (2.5, 7.2), (2.0, 6.9), (2.4, 7.0), (2.7, 7.0)]
for dx, dy in bow_dots:
    ax.plot(dx, dy, 'o', color='#8a6878', markersize=4, zorder=6)

# ============================================
# 6. LEFT EYE - Brown almond shape
# ============================================
left_eye_white = patches.Ellipse((3.5, 5.0), 1.0, 0.75, linewidth=0, facecolor='#fffcf5', zorder=5)
ax.add_patch(left_eye_white)
left_eye_outline = patches.Ellipse((3.5, 5.0), 1.05, 0.8, linewidth=1.5, edgecolor='#3a3230', facecolor='none', zorder=6)
ax.add_patch(left_eye_outline)
left_iris = patches.Circle((3.5, 4.95), 0.32, linewidth=0, facecolor='#7a5540', zorder=6)
ax.add_patch(left_iris)
left_pupil = patches.Circle((3.5, 4.95), 0.12, linewidth=0, facecolor='#1a1412', zorder=7)
ax.add_patch(left_pupil)

# ============================================
# 7. RIGHT EYE - Brown almond shape
# ============================================
right_eye_white = patches.Ellipse((6.5, 5.0), 1.0, 0.75, linewidth=0, facecolor='#fffcf5', zorder=5)
ax.add_patch(right_eye_white)
right_eye_outline = patches.Ellipse((6.5, 5.0), 1.05, 0.8, linewidth=1.5, edgecolor='#3a3230', facecolor='none', zorder=6)
ax.add_patch(right_eye_outline)
right_iris = patches.Circle((6.5, 4.95), 0.32, linewidth=0, facecolor='#7a5540', zorder=6)
ax.add_patch(right_iris)
right_pupil = patches.Circle((6.5, 4.95), 0.12, linewidth=0, facecolor='#1a1412', zorder=7)
ax.add_patch(right_pupil)

# ============================================
# 8. NOSE - Simple vertical line on right side
# ============================================
ax.plot([5.4, 5.45, 5.5], [4.4, 3.9, 3.7], color='#d4b090', linewidth=2, zorder=5)

# ============================================
# 9. MOUTH - Red lips (horizontal ellipse shape like original)
# ============================================
# The original image has a more horizontal/elliptical mouth shape
# with slight smile at corners

# Main mouth shape - ellipse
mouth = patches.Ellipse((5, 3.0), 2.2, 0.6, linewidth=0, facecolor='#e02020', zorder=5)
ax.add_patch(mouth)

# Add smile corners - small triangular curves at edges
# Left corner going up
ax.plot([3.85, 3.6], [3.0, 3.15], color='#e02020', linewidth=4, zorder=5)
# Right corner going up  
ax.plot([6.15, 6.4], [3.0, 3.15], color='#e02020', linewidth=4, zorder=5)

# Darker line through middle of lips
ax.plot([4.0, 6.0], [3.0, 3.0], color='#8a1515', linewidth=1, zorder=6)

# ============================================
# 10. NECK - Short cream connector
# ============================================
neck = patches.Rectangle((4.0, 1.2), 2.0, 1.3, linewidth=0, facecolor='#f5e2c8', zorder=3)
neck.set_clip_path(clip_circle)
ax.add_patch(neck)

# ============================================
# 11. PINK SHIRT - Light pink with curved neckline
# ============================================
shirt_x = [1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 9.0, 1.0, 1.0]
shirt_y = [1.5, 1.8, 2.0, 1.7, 1.4, 1.2, 1.4, 1.7, 2.0, 1.8, 1.5, 0.0, 0.0, 1.5]
shirt_patch = plt.Polygon(list(zip(shirt_x, shirt_y)), facecolor='#e8c8d8', zorder=3)
shirt_patch.set_clip_path(clip_circle)
ax.add_patch(shirt_patch)

# Neckline dark outline
ax.plot([3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0], [2.0, 1.7, 1.4, 1.2, 1.4, 1.7, 2.0], 
        color='#2a2320', linewidth=2, zorder=4)

# ============================================
# Hair strands over face sides
# ============================================
# Left strand
left_strand_x = [1.8, 2.0, 2.3, 2.5, 2.3, 2.0, 1.8]
left_strand_y = [5.5, 4.5, 3.5, 2.5, 3.5, 4.5, 5.5]
left_strand = plt.Polygon(list(zip(left_strand_x, left_strand_y)), facecolor='#2a2320', zorder=4)
left_strand.set_clip_path(clip_circle)
ax.add_patch(left_strand)

# Right strand
right_strand_x = [8.2, 8.0, 7.7, 7.5, 7.7, 8.0, 8.2]
right_strand_y = [5.5, 4.5, 3.5, 2.5, 3.5, 4.5, 5.5]
right_strand = plt.Polygon(list(zip(right_strand_x, right_strand_y)), facecolor='#2a2320', zorder=4)
right_strand.set_clip_path(clip_circle)
ax.add_patch(right_strand)

plt.tight_layout()
plt.savefig('she_output.png', facecolor='#0a0a0a', dpi=150, bbox_inches='tight', pad_inches=0.1)
plt.show()
