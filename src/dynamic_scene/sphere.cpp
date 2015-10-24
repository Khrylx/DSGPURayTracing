#include "sphere.h"

#include "../static_scene/object.h"
#include "../misc/sphere_drawing.h"

namespace CMU462 { namespace DynamicScene {

Sphere::Sphere(const Collada::SphereInfo& info, 
               const Vector3D& position, const double scale) : 
  p(position), r(info.radius * scale) { 
  if (info.material) {
    bsdf = info.material->bsdf;
  } else {
    bsdf = new DiffuseBSDF(Spectrum(0.5f,0.5f,0.5f));    
  }
}

void Sphere::set_draw_styles(DrawStyle *defaultStyle, DrawStyle *hoveredStyle,
                             DrawStyle *selectedStyle) {
  style = defaultStyle;
}

void Sphere::render_in_opengl() const {
  Misc::draw_sphere_opengl(p, r);
}

BBox Sphere::get_bbox() {
  return BBox(p.x - r, p.y - r, p.z - r, p.x + r, p.y + r, p.z + r);
}

BSDF* Sphere::get_bsdf() {
  return bsdf;
}

StaticScene::SceneObject *Sphere::get_static_object() {
  return new StaticScene::SphereObject(p, r, bsdf);
}

} // namespace DynamicScene
} // namespace CMU462
