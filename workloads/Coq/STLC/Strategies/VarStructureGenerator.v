From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Inductive Typ_leaf_ctor :=
  | Ctor_leaf_TBool.

Inductive Expr_leaf_ctor :=
  | Ctor_leaf_Var
  | Ctor_leaf_Bool.

Inductive Typ_ctor :=
  | Ctor_TBool
  | Ctor_TFun.

Inductive Expr_ctor :=
  | Ctor_Var
  | Ctor_Bool
  | Ctor_Abs
  | Ctor_App.

Inductive Typ_ctor_Typ_ctor :=
  | MkTyp_ctor_Typ_ctor : Typ_ctor -> Typ_ctor -> Typ_ctor_Typ_ctor.
Inductive Expr_ctor_Expr_ctor :=
  | MkExpr_ctor_Expr_ctor : Expr_ctor -> Expr_ctor -> Expr_ctor_Expr_ctor.
Inductive Expr_leaf_ctor_Expr_leaf_ctor :=
  | MkExpr_leaf_ctor_Expr_leaf_ctor : Expr_leaf_ctor -> Expr_leaf_ctor -> Expr_leaf_ctor_Expr_leaf_ctor.
Inductive Typ_ctor_Expr_ctor :=
  | MkTyp_ctor_Expr_ctor : Typ_ctor -> Expr_ctor -> Typ_ctor_Expr_ctor.
Inductive Typ_ctor_Expr_leaf_ctor :=
  | MkTyp_ctor_Expr_leaf_ctor : Typ_ctor -> Expr_leaf_ctor -> Typ_ctor_Expr_leaf_ctor.
Inductive Typ_leaf_ctor_Typ_leaf_ctor :=
  | MkTyp_leaf_ctor_Typ_leaf_ctor : Typ_leaf_ctor -> Typ_leaf_ctor -> Typ_leaf_ctor_Typ_leaf_ctor.
Definition gen_leaf_Typ (chosen_ctor : Typ_leaf_ctor) (stack1 : nat) (stack2 : nat) : G Typ :=
  match chosen_ctor with
  | Ctor_leaf_TBool =>
    returnGen (TBool )
  end.

Definition gen_leaf_Expr (chosen_ctor : Expr_leaf_ctor) (stack1 : nat) (stack2 : nat) : G Expr :=
  match chosen_ctor with
  | Ctor_leaf_Var =>
    let weight_1 :=
    match ((stack1 : nat), (stack2 : nat)) with 
    | (4, 4) => 50
    | (4, 5) => 50
    | (4, 6) => 50
    | (5, 4) => 50
    | (5, 5) => 50
    | (5, 6) => 50
    | (6, 4) => 50
    | (6, 5) => 50
    | (6, 6) => 50
    | _ => 500
    end
    in
    bindGen (freq [
      (weight_1, returnGen 1);
      (100-weight_1, returnGen 0)
    ]) (fun n1 =>
    let weight_2 :=
    match ((stack1 : nat), (stack2 : nat)) with 
    | (4, 4) => 50
    | (4, 5) => 50
    | (4, 6) => 50
    | (5, 4) => 50
    | (5, 5) => 50
    | (5, 6) => 50
    | (6, 4) => 50
    | (6, 5) => 50
    | (6, 6) => 50
    | _ => 500
    end
    in
    bindGen (freq [
      (weight_2, returnGen 2);
      (100-weight_2, returnGen 0)
    ]) (fun n2 =>
    let weight_4 :=
    match ((stack1 : nat), (stack2 : nat)) with 
    | (4, 4) => 50
    | (4, 5) => 50
    | (4, 6) => 50
    | (5, 4) => 50
    | (5, 5) => 50
    | (5, 6) => 50
    | (6, 4) => 50
    | (6, 5) => 50
    | (6, 6) => 50
    | _ => 500
    end
    in
    bindGen (freq [
      (weight_4, returnGen 4);
      (100-weight_4, returnGen 0)
    ]) (fun n4 =>
    let weight_8 :=
    match ((stack1 : nat), (stack2 : nat)) with 
    | (4, 4) => 50
    | (4, 5) => 50
    | (4, 6) => 50
    | (5, 4) => 50
    | (5, 5) => 50
    | (5, 6) => 50
    | (6, 4) => 50
    | (6, 5) => 50
    | (6, 6) => 50
    | _ => 500
    end
    in
    bindGen (freq [
      (weight_8, returnGen 8);
      (100-weight_8, returnGen 0)
    ]) (fun n8 =>
    let weight_16 :=
    match ((stack1 : nat), (stack2 : nat)) with 
    | (4, 4) => 50
    | (4, 5) => 50
    | (4, 6) => 50
    | (5, 4) => 50
    | (5, 5) => 50
    | (5, 6) => 50
    | (6, 4) => 50
    | (6, 5) => 50
    | (6, 6) => 50
    | _ => 500
    end
    in
    bindGen (freq [
      (weight_16, returnGen 16);
      (100-weight_16, returnGen 0)
    ]) (fun n16 =>
    let weight_32 :=
    match ((stack1 : nat), (stack2 : nat)) with 
    | (4, 4) => 50
    | (4, 5) => 50
    | (4, 6) => 50
    | (5, 4) => 50
    | (5, 5) => 50
    | (5, 6) => 50
    | (6, 4) => 50
    | (6, 5) => 50
    | (6, 6) => 50
    | _ => 500
    end
    in
    bindGen (freq [
      (weight_32, returnGen 32);
      (100-weight_32, returnGen 0)
    ]) (fun n32 =>
    let p1 := n1 + n2 + n4 + n8 + n16 + n32 in 
    returnGen (Var p1)))))))
  | Ctor_leaf_Bool =>
    let weight_true :=
    match ((stack1 : nat), (stack2 : nat)) with 
    | (4, 4) => 50
    | (4, 5) => 50
    | (4, 6) => 50
    | (5, 4) => 50
    | (5, 5) => 50
    | (5, 6) => 50
    | (6, 4) => 50
    | (6, 5) => 50
    | (6, 6) => 50
    | _ => 500
    end
    in
    bindGen (freq [
      (weight_true, returnGen true);
      (100-weight_true, returnGen false)
    ]) (fun p1 : bool =>
    returnGen (Bool p1))
  end.

Fixpoint gen_Typ (size : nat) (chosen_ctor : Typ_ctor) (stack1 : nat) (stack2 : nat) : G Typ :=
  match size with
  | 0 =>
    match chosen_ctor with
    | Ctor_TBool =>
      returnGen (TBool )
    | Ctor_TFun =>
      bindGen (freq [
        (* no alternatives, so lets just put this again *)
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (3, 1)) => 50
          | (0, (3, 2)) => 50
          | _ => 500
          end,
          returnGen (MkTyp_leaf_ctor_Typ_leaf_ctor Ctor_leaf_TBool Ctor_leaf_TBool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (3, 1)) => 50
          | (0, (3, 2)) => 50
          | _ => 500
          end,
          returnGen (MkTyp_leaf_ctor_Typ_leaf_ctor Ctor_leaf_TBool Ctor_leaf_TBool)
        )
      ]) (fun param_variantis =>
      let '(MkTyp_leaf_ctor_Typ_leaf_ctor param1_ctor param2_ctor) := param_variantis in
      bindGen (gen_leaf_Typ param1_ctor (stack2 : nat) 1)
      (fun p1 : Typ =>
      bindGen (gen_leaf_Typ param2_ctor (stack2 : nat) 2)
      (fun p2 : Typ =>
      returnGen (TFun p1 p2))))
  end
  | S size' =>
    match chosen_ctor with
    | Ctor_TBool =>
      returnGen (TBool )
    | Ctor_TFun =>
      bindGen (freq [
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (0, 3)) => 50
          | (1, (4, 3)) => 50
          | (1, (5, 3)) => 50
          | (1, (6, 3)) => 50
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Typ_ctor Ctor_TBool Ctor_TBool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (0, 3)) => 50
          | (1, (4, 3)) => 50
          | (1, (5, 3)) => 50
          | (1, (6, 3)) => 50
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Typ_ctor Ctor_TFun Ctor_TBool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (0, 3)) => 50
          | (1, (4, 3)) => 50
          | (1, (5, 3)) => 50
          | (1, (6, 3)) => 50
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Typ_ctor Ctor_TBool Ctor_TFun)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (0, 3)) => 50
          | (1, (4, 3)) => 50
          | (1, (5, 3)) => 50
          | (1, (6, 3)) => 50
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Typ_ctor Ctor_TFun Ctor_TFun)
        )
      ]) (fun param_variantis =>
      let '(MkTyp_ctor_Typ_ctor param1_ctor param2_ctor) := param_variantis in
      bindGen (gen_Typ size' param1_ctor (stack2 : nat) 1)
      (fun p1 : Typ =>
      bindGen (gen_Typ size' param2_ctor (stack2 : nat) 2)
      (fun p2 : Typ =>
      returnGen (TFun p1 p2))))
  end
  end.

Fixpoint gen_Expr (size : nat) (chosen_ctor : Expr_ctor) (stack1 : nat) (stack2 : nat) : G Expr :=
  match size with
  | 0 =>
    match chosen_ctor with
    | Ctor_Var =>
      let weight_1 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (4, 4)) => 50
      | (0, (4, 5)) => 50
      | (0, (4, 6)) => 50
      | (0, (5, 4)) => 50
      | (0, (5, 5)) => 50
      | (0, (5, 6)) => 50
      | (0, (6, 4)) => 50
      | (0, (6, 5)) => 50
      | (0, (6, 6)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_1, returnGen 1);
        (100-weight_1, returnGen 0)
      ]) (fun n1 =>
      let weight_2 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (4, 4)) => 50
      | (0, (4, 5)) => 50
      | (0, (4, 6)) => 50
      | (0, (5, 4)) => 50
      | (0, (5, 5)) => 50
      | (0, (5, 6)) => 50
      | (0, (6, 4)) => 50
      | (0, (6, 5)) => 50
      | (0, (6, 6)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_2, returnGen 2);
        (100-weight_2, returnGen 0)
      ]) (fun n2 =>
      let weight_4 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (4, 4)) => 50
      | (0, (4, 5)) => 50
      | (0, (4, 6)) => 50
      | (0, (5, 4)) => 50
      | (0, (5, 5)) => 50
      | (0, (5, 6)) => 50
      | (0, (6, 4)) => 50
      | (0, (6, 5)) => 50
      | (0, (6, 6)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_4, returnGen 4);
        (100-weight_4, returnGen 0)
      ]) (fun n4 =>
      let weight_8 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (4, 4)) => 50
      | (0, (4, 5)) => 50
      | (0, (4, 6)) => 50
      | (0, (5, 4)) => 50
      | (0, (5, 5)) => 50
      | (0, (5, 6)) => 50
      | (0, (6, 4)) => 50
      | (0, (6, 5)) => 50
      | (0, (6, 6)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_8, returnGen 8);
        (100-weight_8, returnGen 0)
      ]) (fun n8 =>
      let weight_16 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (4, 4)) => 50
      | (0, (4, 5)) => 50
      | (0, (4, 6)) => 50
      | (0, (5, 4)) => 50
      | (0, (5, 5)) => 50
      | (0, (5, 6)) => 50
      | (0, (6, 4)) => 50
      | (0, (6, 5)) => 50
      | (0, (6, 6)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_16, returnGen 16);
        (100-weight_16, returnGen 0)
      ]) (fun n16 =>
      let weight_32 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (4, 4)) => 50
      | (0, (4, 5)) => 50
      | (0, (4, 6)) => 50
      | (0, (5, 4)) => 50
      | (0, (5, 5)) => 50
      | (0, (5, 6)) => 50
      | (0, (6, 4)) => 50
      | (0, (6, 5)) => 50
      | (0, (6, 6)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_32, returnGen 32);
        (100-weight_32, returnGen 0)
      ]) (fun n32 =>
      let p1 := n1 + n2 + n4 + n8 + n16 + n32 in 
      returnGen (Var p1)))))))
    | Ctor_Bool =>
      let weight_true :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (4, 4)) => 50
      | (0, (4, 5)) => 50
      | (0, (4, 6)) => 50
      | (0, (5, 4)) => 50
      | (0, (5, 5)) => 50
      | (0, (5, 6)) => 50
      | (0, (6, 4)) => 50
      | (0, (6, 5)) => 50
      | (0, (6, 6)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_true, returnGen true);
        (100-weight_true, returnGen false)
      ]) (fun p1 : bool =>
      returnGen (Bool p1))
    | Ctor_Abs =>
      bindGen (freq [
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (4, 4)) => 2
          | (0, (4, 5)) => 5
          | (0, (4, 6)) => 5
          | (0, (5, 4)) => 20
          | (0, (5, 5)) => 37
          | (0, (5, 6)) => 34
          | (0, (6, 4)) => 18
          | (0, (6, 5)) => 31
          | (0, (6, 6)) => 32
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_leaf_ctor Ctor_TBool Ctor_leaf_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (4, 4)) => 2
          | (0, (4, 5)) => 5
          | (0, (4, 6)) => 5
          | (0, (5, 4)) => 20
          | (0, (5, 5)) => 37
          | (0, (5, 6)) => 34
          | (0, (6, 4)) => 18
          | (0, (6, 5)) => 31
          | (0, (6, 6)) => 32
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_leaf_ctor Ctor_TFun Ctor_leaf_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (4, 4)) => 82
          | (0, (4, 5)) => 79
          | (0, (4, 6)) => 78
          | (0, (5, 4)) => 70
          | (0, (5, 5)) => 61
          | (0, (5, 6)) => 62
          | (0, (6, 4)) => 71
          | (0, (6, 5)) => 64
          | (0, (6, 6)) => 63
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_leaf_ctor Ctor_TBool Ctor_leaf_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (4, 4)) => 82
          | (0, (4, 5)) => 79
          | (0, (4, 6)) => 78
          | (0, (5, 4)) => 70
          | (0, (5, 5)) => 61
          | (0, (5, 6)) => 62
          | (0, (6, 4)) => 71
          | (0, (6, 5)) => 64
          | (0, (6, 6)) => 63
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_leaf_ctor Ctor_TFun Ctor_leaf_Bool)
        )
      ]) (fun param_variantis =>
      let '(MkTyp_ctor_Expr_leaf_ctor param1_ctor param2_ctor) := param_variantis in
      bindGen (gen_Typ 1 param1_ctor (stack2 : nat) 3)
      (fun p1 : Typ =>
      bindGen (gen_leaf_Expr param2_ctor (stack2 : nat) 4)
      (fun p2 : Expr =>
      returnGen (Abs p1 p2))))
    | Ctor_App =>
      bindGen (freq [
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (4, 4)) => 2
          | (0, (4, 5)) => 7
          | (0, (4, 6)) => 6
          | (0, (5, 4)) => 30
          | (0, (5, 5)) => 39
          | (0, (5, 6)) => 40
          | (0, (6, 4)) => 29
          | (0, (6, 5)) => 38
          | (0, (6, 6)) => 38
          | _ => 500
          end,
          returnGen (MkExpr_leaf_ctor_Expr_leaf_ctor Ctor_leaf_Var Ctor_leaf_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (4, 4)) => 2
          | (0, (4, 5)) => 8
          | (0, (4, 6)) => 6
          | (0, (5, 4)) => 32
          | (0, (5, 5)) => 39
          | (0, (5, 6)) => 40
          | (0, (6, 4)) => 29
          | (0, (6, 5)) => 38
          | (0, (6, 6)) => 38
          | _ => 500
          end,
          returnGen (MkExpr_leaf_ctor_Expr_leaf_ctor Ctor_leaf_Bool Ctor_leaf_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (4, 4)) => 2
          | (0, (4, 5)) => 9
          | (0, (4, 6)) => 6
          | (0, (5, 4)) => 32
          | (0, (5, 5)) => 40
          | (0, (5, 6)) => 41
          | (0, (6, 4)) => 30
          | (0, (6, 5)) => 38
          | (0, (6, 6)) => 38
          | _ => 500
          end,
          returnGen (MkExpr_leaf_ctor_Expr_leaf_ctor Ctor_leaf_Var Ctor_leaf_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (4, 4)) => 93
          | (0, (4, 5)) => 90
          | (0, (4, 6)) => 91
          | (0, (5, 4)) => 79
          | (0, (5, 5)) => 71
          | (0, (5, 6)) => 69
          | (0, (6, 4)) => 80
          | (0, (6, 5)) => 73
          | (0, (6, 6)) => 73
          | _ => 500
          end,
          returnGen (MkExpr_leaf_ctor_Expr_leaf_ctor Ctor_leaf_Bool Ctor_leaf_Bool)
        )
      ]) (fun param_variantis =>
      let '(MkExpr_leaf_ctor_Expr_leaf_ctor param1_ctor param2_ctor) := param_variantis in
      bindGen (gen_leaf_Expr param1_ctor (stack2 : nat) 5)
      (fun p1 : Expr =>
      bindGen (gen_leaf_Expr param2_ctor (stack2 : nat) 6)
      (fun p2 : Expr =>
      returnGen (App p1 p2))))
  end
  | S size' =>
    match chosen_ctor with
    | Ctor_Var =>
      let weight_1 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (4, 4)) => 50
      | (1, (4, 5)) => 50
      | (1, (4, 6)) => 50
      | (1, (5, 4)) => 50
      | (1, (5, 5)) => 50
      | (1, (5, 6)) => 50
      | (1, (6, 4)) => 50
      | (1, (6, 5)) => 50
      | (1, (6, 6)) => 50
      | (2, (4, 4)) => 50
      | (2, (4, 5)) => 50
      | (2, (4, 6)) => 50
      | (2, (5, 4)) => 50
      | (2, (5, 5)) => 50
      | (2, (5, 6)) => 50
      | (2, (6, 4)) => 50
      | (2, (6, 5)) => 50
      | (2, (6, 6)) => 50
      | (3, (0, 4)) => 50
      | (3, (0, 5)) => 50
      | (3, (0, 6)) => 50
      | (4, (0, 0)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_1, returnGen 1);
        (100-weight_1, returnGen 0)
      ]) (fun n1 =>
      let weight_2 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (4, 4)) => 50
      | (1, (4, 5)) => 50
      | (1, (4, 6)) => 50
      | (1, (5, 4)) => 50
      | (1, (5, 5)) => 50
      | (1, (5, 6)) => 50
      | (1, (6, 4)) => 50
      | (1, (6, 5)) => 50
      | (1, (6, 6)) => 50
      | (2, (4, 4)) => 50
      | (2, (4, 5)) => 50
      | (2, (4, 6)) => 50
      | (2, (5, 4)) => 50
      | (2, (5, 5)) => 50
      | (2, (5, 6)) => 50
      | (2, (6, 4)) => 50
      | (2, (6, 5)) => 50
      | (2, (6, 6)) => 50
      | (3, (0, 4)) => 50
      | (3, (0, 5)) => 50
      | (3, (0, 6)) => 50
      | (4, (0, 0)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_2, returnGen 2);
        (100-weight_2, returnGen 0)
      ]) (fun n2 =>
      let weight_4 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (4, 4)) => 50
      | (1, (4, 5)) => 50
      | (1, (4, 6)) => 50
      | (1, (5, 4)) => 50
      | (1, (5, 5)) => 50
      | (1, (5, 6)) => 50
      | (1, (6, 4)) => 50
      | (1, (6, 5)) => 50
      | (1, (6, 6)) => 50
      | (2, (4, 4)) => 50
      | (2, (4, 5)) => 50
      | (2, (4, 6)) => 50
      | (2, (5, 4)) => 50
      | (2, (5, 5)) => 50
      | (2, (5, 6)) => 50
      | (2, (6, 4)) => 50
      | (2, (6, 5)) => 50
      | (2, (6, 6)) => 50
      | (3, (0, 4)) => 50
      | (3, (0, 5)) => 50
      | (3, (0, 6)) => 50
      | (4, (0, 0)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_4, returnGen 4);
        (100-weight_4, returnGen 0)
      ]) (fun n4 =>
      let weight_8 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (4, 4)) => 50
      | (1, (4, 5)) => 50
      | (1, (4, 6)) => 50
      | (1, (5, 4)) => 50
      | (1, (5, 5)) => 50
      | (1, (5, 6)) => 50
      | (1, (6, 4)) => 50
      | (1, (6, 5)) => 50
      | (1, (6, 6)) => 50
      | (2, (4, 4)) => 50
      | (2, (4, 5)) => 50
      | (2, (4, 6)) => 50
      | (2, (5, 4)) => 50
      | (2, (5, 5)) => 50
      | (2, (5, 6)) => 50
      | (2, (6, 4)) => 50
      | (2, (6, 5)) => 50
      | (2, (6, 6)) => 50
      | (3, (0, 4)) => 50
      | (3, (0, 5)) => 50
      | (3, (0, 6)) => 50
      | (4, (0, 0)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_8, returnGen 8);
        (100-weight_8, returnGen 0)
      ]) (fun n8 =>
      let weight_16 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (4, 4)) => 50
      | (1, (4, 5)) => 50
      | (1, (4, 6)) => 50
      | (1, (5, 4)) => 50
      | (1, (5, 5)) => 50
      | (1, (5, 6)) => 50
      | (1, (6, 4)) => 50
      | (1, (6, 5)) => 50
      | (1, (6, 6)) => 50
      | (2, (4, 4)) => 50
      | (2, (4, 5)) => 50
      | (2, (4, 6)) => 50
      | (2, (5, 4)) => 50
      | (2, (5, 5)) => 50
      | (2, (5, 6)) => 50
      | (2, (6, 4)) => 50
      | (2, (6, 5)) => 50
      | (2, (6, 6)) => 50
      | (3, (0, 4)) => 50
      | (3, (0, 5)) => 50
      | (3, (0, 6)) => 50
      | (4, (0, 0)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_16, returnGen 16);
        (100-weight_16, returnGen 0)
      ]) (fun n16 =>
      let weight_32 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (4, 4)) => 50
      | (1, (4, 5)) => 50
      | (1, (4, 6)) => 50
      | (1, (5, 4)) => 50
      | (1, (5, 5)) => 50
      | (1, (5, 6)) => 50
      | (1, (6, 4)) => 50
      | (1, (6, 5)) => 50
      | (1, (6, 6)) => 50
      | (2, (4, 4)) => 50
      | (2, (4, 5)) => 50
      | (2, (4, 6)) => 50
      | (2, (5, 4)) => 50
      | (2, (5, 5)) => 50
      | (2, (5, 6)) => 50
      | (2, (6, 4)) => 50
      | (2, (6, 5)) => 50
      | (2, (6, 6)) => 50
      | (3, (0, 4)) => 50
      | (3, (0, 5)) => 50
      | (3, (0, 6)) => 50
      | (4, (0, 0)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_32, returnGen 32);
        (100-weight_32, returnGen 0)
      ]) (fun n32 =>
      let p1 := n1 + n2 + n4 + n8 + n16 + n32 in 
      returnGen (Var p1)))))))
    | Ctor_Bool =>
      let weight_true :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (4, 4)) => 50
      | (1, (4, 5)) => 50
      | (1, (4, 6)) => 50
      | (1, (5, 4)) => 50
      | (1, (5, 5)) => 50
      | (1, (5, 6)) => 50
      | (1, (6, 4)) => 50
      | (1, (6, 5)) => 50
      | (1, (6, 6)) => 50
      | (2, (4, 4)) => 50
      | (2, (4, 5)) => 50
      | (2, (4, 6)) => 50
      | (2, (5, 4)) => 50
      | (2, (5, 5)) => 50
      | (2, (5, 6)) => 50
      | (2, (6, 4)) => 50
      | (2, (6, 5)) => 50
      | (2, (6, 6)) => 50
      | (3, (0, 4)) => 50
      | (3, (0, 5)) => 50
      | (3, (0, 6)) => 50
      | (4, (0, 0)) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_true, returnGen true);
        (100-weight_true, returnGen false)
      ]) (fun p1 : bool =>
      returnGen (Bool p1))
    | Ctor_Abs =>
      bindGen (freq [
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 1
          | (1, (4, 5)) => 19
          | (1, (4, 6)) => 19
          | (1, (5, 4)) => 21
          | (1, (5, 5)) => 40
          | (1, (5, 6)) => 39
          | (1, (6, 4)) => 20
          | (1, (6, 5)) => 38
          | (1, (6, 6)) => 38
          | (2, (4, 4)) => 2
          | (2, (4, 5)) => 13
          | (2, (4, 6)) => 14
          | (2, (5, 4)) => 2
          | (2, (5, 5)) => 28
          | (2, (5, 6)) => 28
          | (2, (6, 4)) => 2
          | (2, (6, 5)) => 24
          | (2, (6, 6)) => 23
          | (3, (0, 4)) => 1
          | (3, (0, 5)) => 1
          | (3, (0, 6)) => 1
          | (4, (0, 0)) => 1
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TBool Ctor_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 1
          | (1, (4, 5)) => 19
          | (1, (4, 6)) => 19
          | (1, (5, 4)) => 21
          | (1, (5, 5)) => 40
          | (1, (5, 6)) => 39
          | (1, (6, 4)) => 20
          | (1, (6, 5)) => 38
          | (1, (6, 6)) => 38
          | (2, (4, 4)) => 2
          | (2, (4, 5)) => 13
          | (2, (4, 6)) => 14
          | (2, (5, 4)) => 2
          | (2, (5, 5)) => 28
          | (2, (5, 6)) => 28
          | (2, (6, 4)) => 2
          | (2, (6, 5)) => 24
          | (2, (6, 6)) => 23
          | (3, (0, 4)) => 1
          | (3, (0, 5)) => 1
          | (3, (0, 6)) => 1
          | (4, (0, 0)) => 1
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TFun Ctor_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 78
          | (1, (4, 5)) => 72
          | (1, (4, 6)) => 73
          | (1, (5, 4)) => 66
          | (1, (5, 5)) => 58
          | (1, (5, 6)) => 57
          | (1, (6, 4)) => 63
          | (1, (6, 5)) => 59
          | (1, (6, 6)) => 59
          | (2, (4, 4)) => 58
          | (2, (4, 5)) => 74
          | (2, (4, 6)) => 74
          | (2, (5, 4)) => 52
          | (2, (5, 5)) => 63
          | (2, (5, 6)) => 64
          | (2, (6, 4)) => 58
          | (2, (6, 5)) => 66
          | (2, (6, 6)) => 68
          | (3, (0, 4)) => 25
          | (3, (0, 5)) => 33
          | (3, (0, 6)) => 31
          | (4, (0, 0)) => 15
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TBool Ctor_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 78
          | (1, (4, 5)) => 72
          | (1, (4, 6)) => 73
          | (1, (5, 4)) => 66
          | (1, (5, 5)) => 58
          | (1, (5, 6)) => 57
          | (1, (6, 4)) => 63
          | (1, (6, 5)) => 59
          | (1, (6, 6)) => 59
          | (2, (4, 4)) => 58
          | (2, (4, 5)) => 74
          | (2, (4, 6)) => 74
          | (2, (5, 4)) => 52
          | (2, (5, 5)) => 63
          | (2, (5, 6)) => 64
          | (2, (6, 4)) => 58
          | (2, (6, 5)) => 66
          | (2, (6, 6)) => 68
          | (3, (0, 4)) => 25
          | (3, (0, 5)) => 33
          | (3, (0, 6)) => 31
          | (4, (0, 0)) => 15
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TFun Ctor_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 67
          | (1, (4, 5)) => 57
          | (1, (4, 6)) => 57
          | (1, (5, 4)) => 57
          | (1, (5, 5)) => 54
          | (1, (5, 6)) => 55
          | (1, (6, 4)) => 61
          | (1, (6, 5)) => 54
          | (1, (6, 6)) => 55
          | (2, (4, 4)) => 81
          | (2, (4, 5)) => 60
          | (2, (4, 6)) => 63
          | (2, (5, 4)) => 83
          | (2, (5, 5)) => 58
          | (2, (5, 6)) => 60
          | (2, (6, 4)) => 80
          | (2, (6, 5)) => 58
          | (2, (6, 6)) => 55
          | (3, (0, 4)) => 90
          | (3, (0, 5)) => 91
          | (3, (0, 6)) => 91
          | (4, (0, 0)) => 90
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TBool Ctor_Abs)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 67
          | (1, (4, 5)) => 57
          | (1, (4, 6)) => 57
          | (1, (5, 4)) => 57
          | (1, (5, 5)) => 54
          | (1, (5, 6)) => 55
          | (1, (6, 4)) => 61
          | (1, (6, 5)) => 54
          | (1, (6, 6)) => 55
          | (2, (4, 4)) => 81
          | (2, (4, 5)) => 60
          | (2, (4, 6)) => 63
          | (2, (5, 4)) => 83
          | (2, (5, 5)) => 58
          | (2, (5, 6)) => 60
          | (2, (6, 4)) => 80
          | (2, (6, 5)) => 58
          | (2, (6, 6)) => 55
          | (3, (0, 4)) => 90
          | (3, (0, 5)) => 91
          | (3, (0, 6)) => 91
          | (4, (0, 0)) => 90
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TFun Ctor_Abs)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 46
          | (1, (4, 5)) => 37
          | (1, (4, 6)) => 36
          | (1, (5, 4)) => 48
          | (1, (5, 5)) => 46
          | (1, (5, 6)) => 47
          | (1, (6, 4)) => 49
          | (1, (6, 5)) => 47
          | (1, (6, 6)) => 46
          | (2, (4, 4)) => 39
          | (2, (4, 5)) => 35
          | (2, (4, 6)) => 29
          | (2, (5, 4)) => 42
          | (2, (5, 5)) => 44
          | (2, (5, 6)) => 41
          | (2, (6, 4)) => 50
          | (2, (6, 5)) => 44
          | (2, (6, 6)) => 45
          | (3, (0, 4)) => 9
          | (3, (0, 5)) => 3
          | (3, (0, 6)) => 6
          | (4, (0, 0)) => 32
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TBool Ctor_App)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 46
          | (1, (4, 5)) => 37
          | (1, (4, 6)) => 36
          | (1, (5, 4)) => 48
          | (1, (5, 5)) => 46
          | (1, (5, 6)) => 47
          | (1, (6, 4)) => 49
          | (1, (6, 5)) => 47
          | (1, (6, 6)) => 46
          | (2, (4, 4)) => 39
          | (2, (4, 5)) => 35
          | (2, (4, 6)) => 29
          | (2, (5, 4)) => 42
          | (2, (5, 5)) => 44
          | (2, (5, 6)) => 41
          | (2, (6, 4)) => 50
          | (2, (6, 5)) => 44
          | (2, (6, 6)) => 45
          | (3, (0, 4)) => 9
          | (3, (0, 5)) => 3
          | (3, (0, 6)) => 6
          | (4, (0, 0)) => 32
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TFun Ctor_App)
        )
      ]) (fun param_variantis =>
      let '(MkTyp_ctor_Expr_ctor param1_ctor param2_ctor) := param_variantis in
      bindGen (gen_Typ 1 param1_ctor (stack2 : nat) 3)
      (fun p1 : Typ =>
      bindGen (gen_Expr size' param2_ctor (stack2 : nat) 4)
      (fun p2 : Expr =>
      returnGen (Abs p1 p2))))
    | Ctor_App =>
      bindGen (freq [
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 7
          | (1, (4, 5)) => 40
          | (1, (4, 6)) => 39
          | (1, (5, 4)) => 40
          | (1, (5, 5)) => 48
          | (1, (5, 6)) => 47
          | (1, (6, 4)) => 41
          | (1, (6, 5)) => 47
          | (1, (6, 6)) => 47
          | (2, (4, 4)) => 22
          | (2, (4, 5)) => 40
          | (2, (4, 6)) => 39
          | (2, (5, 4)) => 36
          | (2, (5, 5)) => 46
          | (2, (5, 6)) => 47
          | (2, (6, 4)) => 32
          | (2, (6, 5)) => 44
          | (2, (6, 6)) => 44
          | (3, (0, 4)) => 8
          | (3, (0, 5)) => 20
          | (3, (0, 6)) => 15
          | (4, (0, 0)) => 1
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Var Ctor_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 8
          | (1, (4, 5)) => 41
          | (1, (4, 6)) => 40
          | (1, (5, 4)) => 41
          | (1, (5, 5)) => 48
          | (1, (5, 6)) => 47
          | (1, (6, 4)) => 42
          | (1, (6, 5)) => 47
          | (1, (6, 6)) => 47
          | (2, (4, 4)) => 23
          | (2, (4, 5)) => 40
          | (2, (4, 6)) => 39
          | (2, (5, 4)) => 36
          | (2, (5, 5)) => 46
          | (2, (5, 6)) => 47
          | (2, (6, 4)) => 34
          | (2, (6, 5)) => 44
          | (2, (6, 6)) => 44
          | (3, (0, 4)) => 8
          | (3, (0, 5)) => 20
          | (3, (0, 6)) => 15
          | (4, (0, 0)) => 1
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Bool Ctor_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 8
          | (1, (4, 5)) => 41
          | (1, (4, 6)) => 40
          | (1, (5, 4)) => 40
          | (1, (5, 5)) => 48
          | (1, (5, 6)) => 47
          | (1, (6, 4)) => 41
          | (1, (6, 5)) => 47
          | (1, (6, 6)) => 47
          | (2, (4, 4)) => 23
          | (2, (4, 5)) => 40
          | (2, (4, 6)) => 39
          | (2, (5, 4)) => 37
          | (2, (5, 5)) => 46
          | (2, (5, 6)) => 47
          | (2, (6, 4)) => 32
          | (2, (6, 5)) => 44
          | (2, (6, 6)) => 44
          | (3, (0, 4)) => 8
          | (3, (0, 5)) => 20
          | (3, (0, 6)) => 15
          | (4, (0, 0)) => 1
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Abs Ctor_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 8
          | (1, (4, 5)) => 41
          | (1, (4, 6)) => 40
          | (1, (5, 4)) => 41
          | (1, (5, 5)) => 48
          | (1, (5, 6)) => 47
          | (1, (6, 4)) => 41
          | (1, (6, 5)) => 47
          | (1, (6, 6)) => 47
          | (2, (4, 4)) => 23
          | (2, (4, 5)) => 40
          | (2, (4, 6)) => 39
          | (2, (5, 4)) => 37
          | (2, (5, 5)) => 46
          | (2, (5, 6)) => 47
          | (2, (6, 4)) => 33
          | (2, (6, 5)) => 44
          | (2, (6, 6)) => 44
          | (3, (0, 4)) => 8
          | (3, (0, 5)) => 20
          | (3, (0, 6)) => 15
          | (4, (0, 0)) => 1
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_App Ctor_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 8
          | (1, (4, 5)) => 41
          | (1, (4, 6)) => 40
          | (1, (5, 4)) => 40
          | (1, (5, 5)) => 48
          | (1, (5, 6)) => 47
          | (1, (6, 4)) => 41
          | (1, (6, 5)) => 48
          | (1, (6, 6)) => 47
          | (2, (4, 4)) => 24
          | (2, (4, 5)) => 40
          | (2, (4, 6)) => 39
          | (2, (5, 4)) => 37
          | (2, (5, 5)) => 46
          | (2, (5, 6)) => 47
          | (2, (6, 4)) => 33
          | (2, (6, 5)) => 44
          | (2, (6, 6)) => 44
          | (3, (0, 4)) => 8
          | (3, (0, 5)) => 20
          | (3, (0, 6)) => 15
          | (4, (0, 0)) => 1
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Var Ctor_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 89
          | (1, (4, 5)) => 72
          | (1, (4, 6)) => 73
          | (1, (5, 4)) => 66
          | (1, (5, 5)) => 57
          | (1, (5, 6)) => 57
          | (1, (6, 4)) => 65
          | (1, (6, 5)) => 60
          | (1, (6, 6)) => 59
          | (2, (4, 4)) => 85
          | (2, (4, 5)) => 77
          | (2, (4, 6)) => 73
          | (2, (5, 4)) => 72
          | (2, (5, 5)) => 63
          | (2, (5, 6)) => 63
          | (2, (6, 4)) => 79
          | (2, (6, 5)) => 67
          | (2, (6, 6)) => 67
          | (3, (0, 4)) => 91
          | (3, (0, 5)) => 88
          | (3, (0, 6)) => 90
          | (4, (0, 0)) => 12
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Bool Ctor_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 85
          | (1, (4, 5)) => 59
          | (1, (4, 6)) => 62
          | (1, (5, 4)) => 60
          | (1, (5, 5)) => 52
          | (1, (5, 6)) => 54
          | (1, (6, 4)) => 60
          | (1, (6, 5)) => 51
          | (1, (6, 6)) => 55
          | (2, (4, 4)) => 76
          | (2, (4, 5)) => 60
          | (2, (4, 6)) => 64
          | (2, (5, 4)) => 66
          | (2, (5, 5)) => 57
          | (2, (5, 6)) => 52
          | (2, (6, 4)) => 67
          | (2, (6, 5)) => 56
          | (2, (6, 6)) => 59
          | (3, (0, 4)) => 88
          | (3, (0, 5)) => 83
          | (3, (0, 6)) => 82
          | (4, (0, 0)) => 86
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Abs Ctor_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 58
          | (1, (4, 5)) => 56
          | (1, (4, 6)) => 53
          | (1, (5, 4)) => 52
          | (1, (5, 5)) => 51
          | (1, (5, 6)) => 54
          | (1, (6, 4)) => 58
          | (1, (6, 5)) => 51
          | (1, (6, 6)) => 52
          | (2, (4, 4)) => 54
          | (2, (4, 5)) => 51
          | (2, (4, 6)) => 53
          | (2, (5, 4)) => 54
          | (2, (5, 5)) => 49
          | (2, (5, 6)) => 50
          | (2, (6, 4)) => 52
          | (2, (6, 5)) => 51
          | (2, (6, 6)) => 52
          | (3, (0, 4)) => 37
          | (3, (0, 5)) => 49
          | (3, (0, 6)) => 45
          | (4, (0, 0)) => 6
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_App Ctor_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 8
          | (1, (4, 5)) => 41
          | (1, (4, 6)) => 40
          | (1, (5, 4)) => 40
          | (1, (5, 5)) => 48
          | (1, (5, 6)) => 47
          | (1, (6, 4)) => 41
          | (1, (6, 5)) => 47
          | (1, (6, 6)) => 47
          | (2, (4, 4)) => 24
          | (2, (4, 5)) => 40
          | (2, (4, 6)) => 39
          | (2, (5, 4)) => 36
          | (2, (5, 5)) => 46
          | (2, (5, 6)) => 47
          | (2, (6, 4)) => 32
          | (2, (6, 5)) => 44
          | (2, (6, 6)) => 44
          | (3, (0, 4)) => 8
          | (3, (0, 5)) => 20
          | (3, (0, 6)) => 15
          | (4, (0, 0)) => 1
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Var Ctor_Abs)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 83
          | (1, (4, 5)) => 63
          | (1, (4, 6)) => 62
          | (1, (5, 4)) => 64
          | (1, (5, 5)) => 56
          | (1, (5, 6)) => 54
          | (1, (6, 4)) => 60
          | (1, (6, 5)) => 54
          | (1, (6, 6)) => 53
          | (2, (4, 4)) => 77
          | (2, (4, 5)) => 64
          | (2, (4, 6)) => 69
          | (2, (5, 4)) => 72
          | (2, (5, 5)) => 58
          | (2, (5, 6)) => 56
          | (2, (6, 4)) => 72
          | (2, (6, 5)) => 61
          | (2, (6, 6)) => 58
          | (3, (0, 4)) => 87
          | (3, (0, 5)) => 81
          | (3, (0, 6)) => 83
          | (4, (0, 0)) => 86
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Bool Ctor_Abs)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 69
          | (1, (4, 5)) => 52
          | (1, (4, 6)) => 55
          | (1, (5, 4)) => 58
          | (1, (5, 5)) => 49
          | (1, (5, 6)) => 53
          | (1, (6, 4)) => 52
          | (1, (6, 5)) => 50
          | (1, (6, 6)) => 51
          | (2, (4, 4)) => 65
          | (2, (4, 5)) => 52
          | (2, (4, 6)) => 53
          | (2, (5, 4)) => 63
          | (2, (5, 5)) => 52
          | (2, (5, 6)) => 54
          | (2, (6, 4)) => 61
          | (2, (6, 5)) => 52
          | (2, (6, 6)) => 54
          | (3, (0, 4)) => 78
          | (3, (0, 5)) => 62
          | (3, (0, 6)) => 68
          | (4, (0, 0)) => 98
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Abs Ctor_Abs)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 34
          | (1, (4, 5)) => 50
          | (1, (4, 6)) => 48
          | (1, (5, 4)) => 51
          | (1, (5, 5)) => 49
          | (1, (5, 6)) => 49
          | (1, (6, 4)) => 48
          | (1, (6, 5)) => 51
          | (1, (6, 6)) => 49
          | (2, (4, 4)) => 39
          | (2, (4, 5)) => 44
          | (2, (4, 6)) => 46
          | (2, (5, 4)) => 47
          | (2, (5, 5)) => 47
          | (2, (5, 6)) => 48
          | (2, (6, 4)) => 43
          | (2, (6, 5)) => 46
          | (2, (6, 6)) => 51
          | (3, (0, 4)) => 24
          | (3, (0, 5)) => 37
          | (3, (0, 6)) => 37
          | (4, (0, 0)) => 7
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_App Ctor_Abs)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 8
          | (1, (4, 5)) => 40
          | (1, (4, 6)) => 40
          | (1, (5, 4)) => 40
          | (1, (5, 5)) => 48
          | (1, (5, 6)) => 47
          | (1, (6, 4)) => 42
          | (1, (6, 5)) => 47
          | (1, (6, 6)) => 47
          | (2, (4, 4)) => 23
          | (2, (4, 5)) => 41
          | (2, (4, 6)) => 39
          | (2, (5, 4)) => 36
          | (2, (5, 5)) => 46
          | (2, (5, 6)) => 47
          | (2, (6, 4)) => 32
          | (2, (6, 5)) => 44
          | (2, (6, 6)) => 44
          | (3, (0, 4)) => 8
          | (3, (0, 5)) => 20
          | (3, (0, 6)) => 15
          | (4, (0, 0)) => 1
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Var Ctor_App)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 70
          | (1, (4, 5)) => 52
          | (1, (4, 6)) => 51
          | (1, (5, 4)) => 51
          | (1, (5, 5)) => 52
          | (1, (5, 6)) => 54
          | (1, (6, 4)) => 51
          | (1, (6, 5)) => 50
          | (1, (6, 6)) => 51
          | (2, (4, 4)) => 56
          | (2, (4, 5)) => 55
          | (2, (4, 6)) => 53
          | (2, (5, 4)) => 52
          | (2, (5, 5)) => 52
          | (2, (5, 6)) => 51
          | (2, (6, 4)) => 57
          | (2, (6, 5)) => 54
          | (2, (6, 6)) => 52
          | (3, (0, 4)) => 45
          | (3, (0, 5)) => 39
          | (3, (0, 6)) => 50
          | (4, (0, 0)) => 12
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Bool Ctor_App)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 50
          | (1, (4, 5)) => 48
          | (1, (4, 6)) => 49
          | (1, (5, 4)) => 52
          | (1, (5, 5)) => 50
          | (1, (5, 6)) => 49
          | (1, (6, 4)) => 54
          | (1, (6, 5)) => 51
          | (1, (6, 6)) => 51
          | (2, (4, 4)) => 45
          | (2, (4, 5)) => 47
          | (2, (4, 6)) => 47
          | (2, (5, 4)) => 42
          | (2, (5, 5)) => 49
          | (2, (5, 6)) => 47
          | (2, (6, 4)) => 47
          | (2, (6, 5)) => 48
          | (2, (6, 6)) => 47
          | (3, (0, 4)) => 25
          | (3, (0, 5)) => 32
          | (3, (0, 6)) => 34
          | (4, (0, 0)) => 9
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Abs Ctor_App)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 24
          | (1, (4, 5)) => 44
          | (1, (4, 6)) => 48
          | (1, (5, 4)) => 50
          | (1, (5, 5)) => 49
          | (1, (5, 6)) => 49
          | (1, (6, 4)) => 52
          | (1, (6, 5)) => 48
          | (1, (6, 6)) => 48
          | (2, (4, 4)) => 31
          | (2, (4, 5)) => 44
          | (2, (4, 6)) => 43
          | (2, (5, 4)) => 43
          | (2, (5, 5)) => 47
          | (2, (5, 6)) => 47
          | (2, (6, 4)) => 44
          | (2, (6, 5)) => 46
          | (2, (6, 6)) => 44
          | (3, (0, 4)) => 13
          | (3, (0, 5)) => 24
          | (3, (0, 6)) => 20
          | (4, (0, 0)) => 1
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_App Ctor_App)
        )
      ]) (fun param_variantis =>
      let '(MkExpr_ctor_Expr_ctor param1_ctor param2_ctor) := param_variantis in
      bindGen (gen_Expr size' param1_ctor (stack2 : nat) 5)
      (fun p1 : Expr =>
      bindGen (gen_Expr size' param2_ctor (stack2 : nat) 6)
      (fun p2 : Expr =>
      returnGen (App p1 p2))))
  end
  end.

Definition gSized :=
  bindGen (freq [
    (
      0,
      returnGen Ctor_Var
    );
    (
      3,
      returnGen Ctor_Bool
    );
    (
      26,
      returnGen Ctor_Abs
    );
    (
      93,
      returnGen Ctor_App
    )
  ]) (fun init_ctor =>
  gen_Expr 4 init_ctor 0 0).

Definition test_prop_SinglePreserve :=
forAll gSized (fun (e: Expr) =>
  prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)

Definition test_prop_MultiPreserve :=
forAll gSized (fun (e: Expr) =>
  prop_MultiPreserve e).

(*! QuickChick test_prop_MultiPreserve. *)
          
