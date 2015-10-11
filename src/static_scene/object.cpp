#include "object.h"

#include <vector>
#include <iostream>
#include <unordered_map>

using std::vector;
using std::unordered_map;

namespace CMU462 { namespace StaticScene {

// Mesh object //

Mesh::Mesh(const HalfedgeMesh& mesh) {

  unordered_map<const Vertex *, int> vertexLabels;
  vector<const Vertex *> verts;

  size_t vertexI = 0;
  for (VertexCIter it = mesh.verticesBegin(); it != mesh.verticesEnd(); it++) {
    const Vertex *v = &*it;
    verts.push_back(v);
    vertexLabels[v] = vertexI;
    vertexI++;
  }

  positions = new Vector3D[vertexI];
  normals   = new Vector3D[vertexI];
  for (int i = 0; i < vertexI; i++) {
    positions[i] = verts[i]->position;
    normals[i]   = verts[i]->normal;
  }

  for (FaceCIter f = mesh.facesBegin(); f != mesh.facesEnd(); f++) {
    HalfedgeCIter h = f->halfedge();
    indices.push_back(vertexLabels[&*h->vertex()]);
    indices.push_back(vertexLabels[&*h->next()->vertex()]);
    indices.push_back(vertexLabels[&*h->next()->next()->vertex()]);
  }

  // FIXME (sky): Use the actual material's BRDF
  brdf = new DiffuseBRDF(Spectrum(1,1,1));

}

vector<Primitive*> Mesh::get_primitives() const {

  vector<Primitive*> primitives;
  size_t num_triangles = indices.size() / 3;
  for (size_t i = 0; i < num_triangles; ++i) {
    Triangle* tri = new Triangle(this, indices[i * 3],
                                       indices[i * 3 + 1],
                                       indices[i * 3 + 2]);
    primitives.push_back(tri);
  }
  return primitives;
}

BRDF* Mesh::get_brdf() const {
  return brdf;
}

// Sphere object //

SphereObject::SphereObject(const Vector3D& o, double r) {

  this->o = o;
  this->r = r;

  // FIXME (sky): Use the actual material's BRDF
  brdf = new DiffuseBRDF(Spectrum(1,1,1));
}

std::vector<Primitive*> SphereObject::get_primitives() const {
  std::vector<Primitive*> primitives;
  primitives.push_back(new Sphere(this,o,r));
  return primitives;
}

BRDF* SphereObject::get_brdf() const {
  return brdf;
}


} // namespace StaticScene
} // namespace CMU462
