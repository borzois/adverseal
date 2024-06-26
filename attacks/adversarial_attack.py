from enum import Enum

from attacks.pgd import pgd_attack
from attacks.fgsm import fgsm_attack


class AttackMethod(Enum):
    PGD = "Projected Gradient Descent"
    FGSM = "Fast Gradient Sign Method"
    CW = "Carlini & Wagner"
    JSMA = "Jacobian-based Saliency Map Attack"


class EnabledAttackMethod(Enum):
    PGD = AttackMethod.PGD.value
    FGSM = AttackMethod.FGSM.value


def adversarial_attack(models, attack_type: AttackMethod, x, accelerator, target_tensor, instance_prompt='a picture', num_steps=1, alpha=2.0/255.0, eps=8.0/255.0):
    print("Performing adversarial attack: {}".format(attack_type.value))
    match attack_type:
        case AttackMethod.PGD:
            return pgd_attack(models, x, accelerator, target_tensor, instance_prompt=instance_prompt, num_steps=num_steps, alpha=alpha, eps=eps)
        case AttackMethod.FGSM:
            return fgsm_attack(models, x, accelerator, target_tensor, instance_prompt=instance_prompt, alpha=alpha)
        case AttackMethod.CW:
            raise NotImplementedError("Attack not implemented: {}".format(attack_type))
        case AttackMethod.JSMA:
            raise NotImplementedError("Attack not implemented: {}".format(attack_type))
        case _:
            raise ValueError("Unknown attack method: {}".format(attack_type))
