#include "light_info.h"

using namespace std;

namespace CMU462 { namespace Collada {

LightInfo::LightInfo() {

  light_type = LightType::NONE;

  color = Color(1,1,1,1);

  position  = Vector3D(0,  0,  0);  // COLLADA defaults
  direction = Vector3D(0,  0, -1);  // COLLADA defaults

  float falloff_deg = 45;
  float falloff_exp = 0.15;

  constant_att = 1;
  linear_att = 0;
  quadratic_att = 0;

}

std::ostream& operator<<(std::ostream& os, const LightInfo& light) {

  os << "LightInfo: " << light.name << " (id:" << light.id << ")";

  os << " [";

    switch (light.light_type) {
      case LightType::NONE:
        os << "type=none";
      case LightType::AMBIENT:
        os << " type=ambient"
           << " color=" << light.color;
        break;
      case LightType::DIRECTIONAL:
        os << " type=directional"
           << " color=" << light.color
           << " direction=" << light.direction;
        break;
      case LightType::POINT:
        os << " type=point"
           << " color=" << light.color
           << " position="  << light.position
           << " constant_att="  << light.constant_att
           << " linear_att="    << light.linear_att
           << " quadratic_att=" << light.quadratic_att;
        break;
      case LightType::SPOT:
        os << " type=spot"
           << " color=" << light.color
           << " position="  << light.position
           << " direction=" << light.direction
           << " falloff_deg=" << light.falloff_deg
           << " falloff_exp=" << light.falloff_exp
           << " constant_att="  << light.constant_att
           << " linear_att="    << light.linear_att
           << " quadratic_att=" << light.quadratic_att;
        break;
    }

  os << " ]";

  return os;
}

} // namespace Collada
} // namespace CMU462
