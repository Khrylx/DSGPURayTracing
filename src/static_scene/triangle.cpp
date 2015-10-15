#include "triangle.h"

#include "CMU462/CMU462.h"
#include "GL/glew.h"

namespace CMU462 { namespace StaticScene {

Triangle::Triangle(const Mesh* mesh, size_t v1, size_t v2, size_t v3) :
    mesh(mesh), v1(v1), v2(v2), v3(v3) { }

BBox Triangle::get_bbox() const {

  // TODO:
  // Compute the bounding box of the triangle.
    BBox bb;
    
    bb.expand(mesh->positions[v1]);
    bb.expand(mesh->positions[v2]);
    bb.expand(mesh->positions[v3]);

  return bb;

}

bool Triangle::intersect(const Ray& r) const {

  // TODO:
  // Implement ray - triangle intersection.

    
    
  return intersect(r,NULL);

}

bool Triangle::intersect(const Ray& r, Intersection *i) const {

  // TODO:
  // Implement ray - triangle intersection.
  // When an intersection takes place, the Intersection data should
  // be updated correspondingly.

    Vector3D e1 = mesh->positions[v2]-mesh->positions[v1];
    Vector3D e2 = mesh->positions[v3]-mesh->positions[v1];
    Vector3D s = r.o - mesh->positions[v1];
    
//    Matrix3x3 M;
//    M.column(0) = e1;
//    M.column(1) = e2;
//    M.column(2) = -r.d;
//    
//    Vector3D x = M.inv()*s;
//    double u = x[0];
//    double v = x[1];
//    double t = x[2];
    double f = dot(cross(e1,r.d),e2);
    if (f == 0) {
        return false;
    }
    
    double u = dot(cross(s,r.d),e2)/f;
    double v = dot(cross(e1,r.d),s)/f;
    double t = dot(cross(e1,-s),e2)/f;
    
    //
    if (!(u >= 0 && v >= 0 && u+v <= 1 && t < r.max_t)) {
        return false;
    }
    
    //cout << u <<" "<< v << " " << t << endl;
    r.max_t = t;
    
    if (i) {
        i->brdf = get_brdf();
        i->t = t;
        i->primitive = this;
        Vector3D n = cross(e1,e2);
        if (dot(r.d,n) > 0) {
            i->n = -n;
        }
        else{
            i->n = n;
        }
    }
    
  return true;
  
}

void Triangle::draw(const Color& c) const {
  glColor4f(c.r, c.g, c.b, c.a);
  glBegin(GL_TRIANGLES);
  glVertex3d(mesh->positions[v1].x,
             mesh->positions[v1].y,
             mesh->positions[v1].z);
  glVertex3d(mesh->positions[v2].x,
             mesh->positions[v2].y,
             mesh->positions[v2].z);
  glVertex3d(mesh->positions[v3].x,
             mesh->positions[v3].y,
             mesh->positions[v3].z);
  glEnd();
}

void Triangle::drawOutline(const Color& c) const {
  glColor4f(c.r, c.g, c.b, c.a);
  glBegin(GL_LINE_LOOP);
  glVertex3d(mesh->positions[v1].x,
             mesh->positions[v1].y,
             mesh->positions[v1].z);
  glVertex3d(mesh->positions[v2].x,
             mesh->positions[v2].y,
             mesh->positions[v2].z);
  glVertex3d(mesh->positions[v3].x,
             mesh->positions[v3].y,
             mesh->positions[v3].z);
  glEnd();
}


} // namespace StaticScene
} // namespace CMU462
