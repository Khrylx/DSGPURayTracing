#include "sphere.h"

#include <cmath>

#include  "../bsdf.h"
#include "../misc/sphere_drawing.h"

namespace CMU462 { namespace StaticScene {

bool Sphere::test(const Ray& r, double& t1, double& t2) const {

  // TODO:
  // Implement ray - sphere intersection test.
  // Return true if there are intersections and writing the
  // smaller of the two intersection times in t1 and the larger in t2.

    Vector3D m = o - r.o;
    double b = dot(m,r.d);
    double c = dot(m, m) - r2;
    double delta = b*b - c;
    if (delta < 0) {
        return false;
    }
    
    t1 = b - sqrt(delta);
    t2 = b + sqrt(delta);
    
    if (t1 >= r.max_t || t2 <= r.min_t ) {
        return false;
    }
    
  return true;

}

bool Sphere::intersect(const Ray& r) const {

  // TODO:
  // Implement ray - sphere intersection.
  // Note that you might want to use the the Sphere::test helper here.

    double tmp;
    
  return test(r, tmp, tmp);

}

bool Sphere::intersect(const Ray& r, Intersection *i) const {

  // TODO:
  // Implement ray - sphere intersection.
  // Note again that you might want to use the the Sphere::test helper here.
  // When an intersection takes place, the Intersection data should be updated
  // correspondingly.

    double t1;
    double t2;
    bool res = test(r, t1, t2);
    if (!res) {
        return false;
    }
    i->bsdf = get_bsdf();
    i->t = t1;
    i->primitive = this;
    Vector3D n = r.o + r.d*t1 - o;
    //cout << n <<endl;
    i->n = n;
    r.max_t = t1;
    
  return true;

}

void Sphere::draw(const Color& c) const {
  Misc::draw_sphere_opengl(o, r, c);
}

void Sphere::drawOutline(const Color& c) const {
    //Misc::draw_sphere_opengl(o, r, c);
}


} // namespace StaticScene
} // namespace CMU462
