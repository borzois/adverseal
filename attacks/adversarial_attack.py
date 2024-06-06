from enum import Enum

from attacks.pgd import pgd_attack
from attacks.cw import cw_attack
from attacks.fgsm import fgsm_attack
from attacks.jsma import jsma_attack


class AttackMethod(Enum):
    PGD = "Projected Gradient Descent"
    FGSM = "Fast Gradient Sign Method"
    CW = "Carlini & Wagner"
    JSMA = "Jacobian-based Saliency Map Attack"


def adversarial_attack(models, attack_type: AttackMethod, x, accelerator, target_tensor, num_steps=1, alpha=2.0/255.0, eps=8.0/255.0):
    print("Performing adversarial attack: {}".format(attack_type.value))
    match attack_type:
        case AttackMethod.PGD:
            return pgd_attack(models, x, accelerator, target_tensor, num_steps, alpha=alpha, eps=eps)
        case AttackMethod.FGSM:
            return fgsm_attack(models, x, accelerator, target_tensor, alpha=alpha)
        case AttackMethod.CW:
            return cw_attack(models, x, accelerator, target_tensor, num_steps)
        case AttackMethod.JSMA:
            return jsma_attack(models, x, accelerator, target_tensor, num_steps)
        case _:
            raise ValueError("Unknown attack method: {}".format(attack_type))
