Require Import ZArith.
From QuickChick Require Import QuickChick.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import QcNotation.
Import MonadNotation.
From Coq Require Import List.
Import ListNotations.

From RBT Require Import Impl Spec.

Inductive Color_leaf_ctor :=
  | Ctor_leaf_R
  | Ctor_leaf_B.

Inductive Tree_leaf_ctor :=
  | Ctor_leaf_E.

Inductive Color_ctor :=
  | Ctor_R
  | Ctor_B.

Inductive Tree_ctor :=
  | Ctor_E
  | Ctor_T.

Inductive Color_ctor_Tree_ctor_Tree_ctor :=
  | MkColor_ctor_Tree_ctor_Tree_ctor : Color_ctor -> Tree_ctor -> Tree_ctor -> Color_ctor_Tree_ctor_Tree_ctor.
Inductive Color_ctor_Tree_leaf_ctor_Tree_leaf_ctor :=
  | MkColor_ctor_Tree_leaf_ctor_Tree_leaf_ctor : Color_ctor -> Tree_leaf_ctor -> Tree_leaf_ctor -> Color_ctor_Tree_leaf_ctor_Tree_leaf_ctor.
Definition gen_leaf_Color (chosen_ctor : Color_leaf_ctor) (stack1 : nat) (stack2 : nat) : G Color :=
  match chosen_ctor with
  | Ctor_leaf_R =>
    returnGen (R )
  | Ctor_leaf_B =>
    returnGen (B )
  end.

Definition gen_leaf_Tree (chosen_ctor : Tree_leaf_ctor) (stack1 : nat) (stack2 : nat) : G Tree :=
  match chosen_ctor with
  | Ctor_leaf_E =>
    returnGen (E )
  end.

Fixpoint gen_Color (size : nat) (chosen_ctor : Color_ctor) (stack1 : nat) (stack2 : nat) : G Color :=
  match size with
  | 0 =>
    match chosen_ctor with
    | Ctor_R =>
      returnGen (R )
    | Ctor_B =>
      returnGen (B )
  end
  | S size' =>
    match chosen_ctor with
    | Ctor_R =>
      returnGen (R )
    | Ctor_B =>
      returnGen (B )
  end
  end.

Fixpoint gen_Tree (size : nat) (chosen_ctor : Tree_ctor) (stack1 : nat) (stack2 : nat) : G Tree :=
  match size with
  | 0 =>
    match chosen_ctor with
    | Ctor_E =>
      returnGen (E )
    | Ctor_T =>
      bindGen (freq [
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (2, 2)) => 50
          | (0, (2, 3)) => 50
          | (0, (3, 2)) => 50
          | (0, (3, 3)) => 50
          | _ => 500
          end,
          returnGen (MkColor_ctor_Tree_leaf_ctor_Tree_leaf_ctor Ctor_R Ctor_leaf_E Ctor_leaf_E)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (2, 2)) => 50
          | (0, (2, 3)) => 50
          | (0, (3, 2)) => 50
          | (0, (3, 3)) => 50
          | _ => 500
          end,
          returnGen (MkColor_ctor_Tree_leaf_ctor_Tree_leaf_ctor Ctor_B Ctor_leaf_E Ctor_leaf_E)
        )
      ]) (fun param_variantis =>
      let '(MkColor_ctor_Tree_leaf_ctor_Tree_leaf_ctor param1_ctor param2_ctor param5_ctor) := param_variantis in
      bindGen (gen_Color 0 param1_ctor (stack2 : nat) 1)
      (fun p1 : Color =>
      bindGen (gen_leaf_Tree param2_ctor (stack2 : nat) 2)
      (fun p2 : Tree =>
      let weight_1 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (2, 2)) => 50
      | (0, (2, 3)) => 50
      | (0, (3, 2)) => 50
      | (0, (3, 3)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_1, returnGen 1%Z);
        (100-weight_1, returnGen 0%Z)
      ]) (fun n1 =>
      let weight_2 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (2, 2)) => 50
      | (0, (2, 3)) => 50
      | (0, (3, 2)) => 50
      | (0, (3, 3)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_2, returnGen 2%Z);
        (100-weight_2, returnGen 0%Z)
      ]) (fun n2 =>
      let weight_4 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (2, 2)) => 50
      | (0, (2, 3)) => 50
      | (0, (3, 2)) => 50
      | (0, (3, 3)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_4, returnGen 4%Z);
        (100-weight_4, returnGen 0%Z)
      ]) (fun n4 =>
      let weight_8 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (2, 2)) => 50
      | (0, (2, 3)) => 50
      | (0, (3, 2)) => 50
      | (0, (3, 3)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_8, returnGen 8%Z);
        (100-weight_8, returnGen 0%Z)
      ]) (fun n8 =>
      let weight_16 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (2, 2)) => 50
      | (0, (2, 3)) => 50
      | (0, (3, 2)) => 50
      | (0, (3, 3)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_16, returnGen 16%Z);
        (100-weight_16, returnGen 0%Z)
      ]) (fun n16 =>
      let weight_32 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (2, 2)) => 50
      | (0, (2, 3)) => 50
      | (0, (3, 2)) => 50
      | (0, (3, 3)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_32, returnGen 32%Z);
        (100-weight_32, returnGen 0%Z)
      ]) (fun n32 =>
      let p3 := (n1 + n2 + n4 + n8 + n16 + n32)%Z in 
      let weight_1 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (2, 2)) => 50
      | (0, (2, 3)) => 50
      | (0, (3, 2)) => 50
      | (0, (3, 3)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_1, returnGen 1%Z);
        (100-weight_1, returnGen 0%Z)
      ]) (fun n1 =>
      let weight_2 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (2, 2)) => 50
      | (0, (2, 3)) => 50
      | (0, (3, 2)) => 50
      | (0, (3, 3)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_2, returnGen 2%Z);
        (100-weight_2, returnGen 0%Z)
      ]) (fun n2 =>
      let weight_4 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (2, 2)) => 50
      | (0, (2, 3)) => 50
      | (0, (3, 2)) => 50
      | (0, (3, 3)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_4, returnGen 4%Z);
        (100-weight_4, returnGen 0%Z)
      ]) (fun n4 =>
      let weight_8 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (2, 2)) => 50
      | (0, (2, 3)) => 50
      | (0, (3, 2)) => 50
      | (0, (3, 3)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_8, returnGen 8%Z);
        (100-weight_8, returnGen 0%Z)
      ]) (fun n8 =>
      let weight_16 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (2, 2)) => 50
      | (0, (2, 3)) => 50
      | (0, (3, 2)) => 50
      | (0, (3, 3)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_16, returnGen 16%Z);
        (100-weight_16, returnGen 0%Z)
      ]) (fun n16 =>
      let weight_32 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (2, 2)) => 50
      | (0, (2, 3)) => 50
      | (0, (3, 2)) => 50
      | (0, (3, 3)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_32, returnGen 32%Z);
        (100-weight_32, returnGen 0%Z)
      ]) (fun n32 =>
      let p4 := (n1 + n2 + n4 + n8 + n16 + n32)%Z in 
      bindGen (gen_leaf_Tree param5_ctor (stack2 : nat) 3)
      (fun p5 : Tree =>
      returnGen (T p1 p2 p3 p4 p5)))))))))))))))))
  end
  | S size' =>
    match chosen_ctor with
    | Ctor_E =>
      returnGen (E )
    | Ctor_T =>
      bindGen (freq [
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (2, 2)) => 50
          | (1, (2, 3)) => 50
          | (1, (3, 2)) => 50
          | (1, (3, 3)) => 50
          | (2, (2, 2)) => 50
          | (2, (2, 3)) => 50
          | (2, (3, 2)) => 50
          | (2, (3, 3)) => 50
          | (3, (0, 2)) => 72
          | (3, (0, 3)) => 78
          | (4, (0, 0)) => 96
          | _ => 500
          end,
          returnGen (MkColor_ctor_Tree_ctor_Tree_ctor Ctor_R Ctor_E Ctor_E)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (2, 2)) => 50
          | (1, (2, 3)) => 50
          | (1, (3, 2)) => 50
          | (1, (3, 3)) => 50
          | (2, (2, 2)) => 50
          | (2, (2, 3)) => 50
          | (2, (3, 2)) => 50
          | (2, (3, 3)) => 50
          | (3, (0, 2)) => 48
          | (3, (0, 3)) => 45
          | (4, (0, 0)) => 96
          | _ => 500
          end,
          returnGen (MkColor_ctor_Tree_ctor_Tree_ctor Ctor_B Ctor_E Ctor_E)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (2, 2)) => 50
          | (1, (2, 3)) => 50
          | (1, (3, 2)) => 50
          | (1, (3, 3)) => 50
          | (2, (2, 2)) => 50
          | (2, (2, 3)) => 50
          | (2, (3, 2)) => 50
          | (2, (3, 3)) => 50
          | (3, (0, 2)) => 45
          | (3, (0, 3)) => 42
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkColor_ctor_Tree_ctor_Tree_ctor Ctor_R Ctor_T Ctor_E)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (2, 2)) => 50
          | (1, (2, 3)) => 50
          | (1, (3, 2)) => 50
          | (1, (3, 3)) => 50
          | (2, (2, 2)) => 50
          | (2, (2, 3)) => 50
          | (2, (3, 2)) => 50
          | (2, (3, 3)) => 50
          | (3, (0, 2)) => 45
          | (3, (0, 3)) => 42
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkColor_ctor_Tree_ctor_Tree_ctor Ctor_B Ctor_T Ctor_E)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (2, 2)) => 50
          | (1, (2, 3)) => 50
          | (1, (3, 2)) => 50
          | (1, (3, 3)) => 50
          | (2, (2, 2)) => 50
          | (2, (2, 3)) => 50
          | (2, (3, 2)) => 50
          | (2, (3, 3)) => 50
          | (3, (0, 2)) => 45
          | (3, (0, 3)) => 42
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkColor_ctor_Tree_ctor_Tree_ctor Ctor_R Ctor_E Ctor_T)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (2, 2)) => 50
          | (1, (2, 3)) => 50
          | (1, (3, 2)) => 50
          | (1, (3, 3)) => 50
          | (2, (2, 2)) => 50
          | (2, (2, 3)) => 50
          | (2, (3, 2)) => 50
          | (2, (3, 3)) => 50
          | (3, (0, 2)) => 45
          | (3, (0, 3)) => 42
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkColor_ctor_Tree_ctor_Tree_ctor Ctor_B Ctor_E Ctor_T)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (2, 2)) => 50
          | (1, (2, 3)) => 50
          | (1, (3, 2)) => 50
          | (1, (3, 3)) => 50
          | (2, (2, 2)) => 50
          | (2, (2, 3)) => 50
          | (2, (3, 2)) => 50
          | (2, (3, 3)) => 50
          | (3, (0, 2)) => 45
          | (3, (0, 3)) => 42
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkColor_ctor_Tree_ctor_Tree_ctor Ctor_R Ctor_T Ctor_T)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (2, 2)) => 50
          | (1, (2, 3)) => 50
          | (1, (3, 2)) => 50
          | (1, (3, 3)) => 50
          | (2, (2, 2)) => 50
          | (2, (2, 3)) => 50
          | (2, (3, 2)) => 50
          | (2, (3, 3)) => 50
          | (3, (0, 2)) => 45
          | (3, (0, 3)) => 42
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkColor_ctor_Tree_ctor_Tree_ctor Ctor_B Ctor_T Ctor_T)
        )
      ]) (fun param_variantis =>
      let '(MkColor_ctor_Tree_ctor_Tree_ctor param1_ctor param2_ctor param5_ctor) := param_variantis in
      bindGen (gen_Color 0 param1_ctor (stack2 : nat) 1)
      (fun p1 : Color =>
      bindGen (gen_Tree size' param2_ctor (stack2 : nat) 2)
      (fun p2 : Tree =>
      let weight_1 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (2, 2)) => 50
      | (1, (2, 3)) => 50
      | (1, (3, 2)) => 50
      | (1, (3, 3)) => 50
      | (2, (2, 2)) => 50
      | (2, (2, 3)) => 50
      | (2, (3, 2)) => 50
      | (2, (3, 3)) => 50
      | (3, (0, 2)) => 49
      | (3, (0, 3)) => 50
      | (4, (0, 0)) => 53
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_1, returnGen 1%Z);
        (100-weight_1, returnGen 0%Z)
      ]) (fun n1 =>
      let weight_2 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (2, 2)) => 50
      | (1, (2, 3)) => 50
      | (1, (3, 2)) => 50
      | (1, (3, 3)) => 50
      | (2, (2, 2)) => 50
      | (2, (2, 3)) => 50
      | (2, (3, 2)) => 50
      | (2, (3, 3)) => 50
      | (3, (0, 2)) => 51
      | (3, (0, 3)) => 55
      | (4, (0, 0)) => 48
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_2, returnGen 2%Z);
        (100-weight_2, returnGen 0%Z)
      ]) (fun n2 =>
      let weight_4 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (2, 2)) => 50
      | (1, (2, 3)) => 50
      | (1, (3, 2)) => 50
      | (1, (3, 3)) => 50
      | (2, (2, 2)) => 50
      | (2, (2, 3)) => 50
      | (2, (3, 2)) => 50
      | (2, (3, 3)) => 50
      | (3, (0, 2)) => 46
      | (3, (0, 3)) => 52
      | (4, (0, 0)) => 48
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_4, returnGen 4%Z);
        (100-weight_4, returnGen 0%Z)
      ]) (fun n4 =>
      let weight_8 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (2, 2)) => 50
      | (1, (2, 3)) => 50
      | (1, (3, 2)) => 50
      | (1, (3, 3)) => 50
      | (2, (2, 2)) => 50
      | (2, (2, 3)) => 50
      | (2, (3, 2)) => 50
      | (2, (3, 3)) => 50
      | (3, (0, 2)) => 42
      | (3, (0, 3)) => 57
      | (4, (0, 0)) => 51
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_8, returnGen 8%Z);
        (100-weight_8, returnGen 0%Z)
      ]) (fun n8 =>
      let weight_16 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (2, 2)) => 50
      | (1, (2, 3)) => 50
      | (1, (3, 2)) => 50
      | (1, (3, 3)) => 50
      | (2, (2, 2)) => 50
      | (2, (2, 3)) => 50
      | (2, (3, 2)) => 50
      | (2, (3, 3)) => 50
      | (3, (0, 2)) => 37
      | (3, (0, 3)) => 62
      | (4, (0, 0)) => 52
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_16, returnGen 16%Z);
        (100-weight_16, returnGen 0%Z)
      ]) (fun n16 =>
      let weight_32 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (2, 2)) => 50
      | (1, (2, 3)) => 50
      | (1, (3, 2)) => 50
      | (1, (3, 3)) => 50
      | (2, (2, 2)) => 50
      | (2, (2, 3)) => 50
      | (2, (3, 2)) => 50
      | (2, (3, 3)) => 50
      | (3, (0, 2)) => 33
      | (3, (0, 3)) => 74
      | (4, (0, 0)) => 43
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_32, returnGen 32%Z);
        (100-weight_32, returnGen 0%Z)
      ]) (fun n32 =>
      let p3 := (n1 + n2 + n4 + n8 + n16 + n32)%Z in 
      let weight_1 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (2, 2)) => 50
      | (1, (2, 3)) => 50
      | (1, (3, 2)) => 50
      | (1, (3, 3)) => 50
      | (2, (2, 2)) => 50
      | (2, (2, 3)) => 50
      | (2, (3, 2)) => 50
      | (2, (3, 3)) => 50
      | (3, (0, 2)) => 48
      | (3, (0, 3)) => 46
      | (4, (0, 0)) => 51
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_1, returnGen 1%Z);
        (100-weight_1, returnGen 0%Z)
      ]) (fun n1 =>
      let weight_2 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (2, 2)) => 50
      | (1, (2, 3)) => 50
      | (1, (3, 2)) => 50
      | (1, (3, 3)) => 50
      | (2, (2, 2)) => 50
      | (2, (2, 3)) => 50
      | (2, (3, 2)) => 50
      | (2, (3, 3)) => 50
      | (3, (0, 2)) => 51
      | (3, (0, 3)) => 48
      | (4, (0, 0)) => 45
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_2, returnGen 2%Z);
        (100-weight_2, returnGen 0%Z)
      ]) (fun n2 =>
      let weight_4 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (2, 2)) => 50
      | (1, (2, 3)) => 50
      | (1, (3, 2)) => 50
      | (1, (3, 3)) => 50
      | (2, (2, 2)) => 50
      | (2, (2, 3)) => 50
      | (2, (3, 2)) => 50
      | (2, (3, 3)) => 50
      | (3, (0, 2)) => 51
      | (3, (0, 3)) => 47
      | (4, (0, 0)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_4, returnGen 4%Z);
        (100-weight_4, returnGen 0%Z)
      ]) (fun n4 =>
      let weight_8 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (2, 2)) => 50
      | (1, (2, 3)) => 50
      | (1, (3, 2)) => 50
      | (1, (3, 3)) => 50
      | (2, (2, 2)) => 50
      | (2, (2, 3)) => 50
      | (2, (3, 2)) => 50
      | (2, (3, 3)) => 50
      | (3, (0, 2)) => 53
      | (3, (0, 3)) => 53
      | (4, (0, 0)) => 49
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_8, returnGen 8%Z);
        (100-weight_8, returnGen 0%Z)
      ]) (fun n8 =>
      let weight_16 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (2, 2)) => 50
      | (1, (2, 3)) => 50
      | (1, (3, 2)) => 50
      | (1, (3, 3)) => 50
      | (2, (2, 2)) => 50
      | (2, (2, 3)) => 50
      | (2, (3, 2)) => 50
      | (2, (3, 3)) => 50
      | (3, (0, 2)) => 46
      | (3, (0, 3)) => 50
      | (4, (0, 0)) => 52
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_16, returnGen 16%Z);
        (100-weight_16, returnGen 0%Z)
      ]) (fun n16 =>
      let weight_32 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (2, 2)) => 50
      | (1, (2, 3)) => 50
      | (1, (3, 2)) => 50
      | (1, (3, 3)) => 50
      | (2, (2, 2)) => 50
      | (2, (2, 3)) => 50
      | (2, (3, 2)) => 50
      | (2, (3, 3)) => 50
      | (3, (0, 2)) => 55
      | (3, (0, 3)) => 54
      | (4, (0, 0)) => 49
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_32, returnGen 32%Z);
        (100-weight_32, returnGen 0%Z)
      ]) (fun n32 =>
      let p4 := (n1 + n2 + n4 + n8 + n16 + n32)%Z in 
      bindGen (gen_Tree size' param5_ctor (stack2 : nat) 3)
      (fun p5 : Tree =>
      returnGen (T p1 p2 p3 p4 p5)))))))))))))))))
  end
  end.

Definition gSized :=
  bindGen (freq [
    (
      0,
      returnGen Ctor_E
    );
    (
      88,
      returnGen Ctor_T
    )
  ]) (fun init_ctor =>
  gen_Tree 4 init_ctor 0 0).

(* --------------------- Tests --------------------- *)

Definition test_prop_InsertValid :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun v =>
        (prop_InsertValid t k v)))).

(*! QuickChick test_prop_InsertValid. *)

Definition test_prop_DeleteValid :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
        prop_DeleteValid t k)).

(*! QuickChick test_prop_DeleteValid. *)

Definition test_prop_InsertPost :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
     forAll arbitrary (fun v =>
        prop_InsertPost t k k' v)))).

(*! QuickChick test_prop_InsertPost. *)

Definition test_prop_DeletePost := 
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
        prop_DeletePost t k k'))).

(*! QuickChick test_prop_DeletePost. *)
    
Definition test_prop_InsertModel :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun v =>
        prop_InsertModel t k v))).

(*! QuickChick test_prop_InsertModel. *)
    
Definition test_prop_DeleteModel :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
            prop_DeleteModel t k)).

(*! QuickChick test_prop_DeleteModel. *)

Definition test_prop_InsertInsert :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v =>
    forAll arbitrary (fun v' =>     
        prop_InsertInsert t k k' v v'))))).

(*! QuickChick test_prop_InsertInsert. *)
    
Definition test_prop_InsertDelete := 
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v =>
        prop_InsertDelete t k k' v)))).

(*! QuickChick test_prop_InsertDelete. *)
    
Definition test_prop_DeleteInsert := 
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v' =>
        prop_DeleteInsert t k k' v')))).

(*! QuickChick test_prop_DeleteInsert. *)
    
Definition test_prop_DeleteDelete :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
        prop_DeleteDelete t k k'))).

(*! QuickChick test_prop_DeleteDelete. *)
          
