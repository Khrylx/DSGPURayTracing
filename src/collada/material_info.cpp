#include "material_info.h"

using namespace std;

namespace CMU462 { namespace Collada {

std::ostream& operator<<(std::ostream& os, const MaterialInfo& material) {

  os << "MaterialInfo: " << material.name << " (id:" << material.id << ")";


  os << " [";

    os << " emit=" << material.Ce;
    os << " ambi=" << material.Ca;
    os << " diff=" << material.Cd;
    os << " spec=" << material.Cs;

    os << " shininess=" << material.Ns;

    os << " refractive_index=" << material.Ni;

  os << " ]";

  return os;
}

} // namespace Collada
} // namespace CMU462
