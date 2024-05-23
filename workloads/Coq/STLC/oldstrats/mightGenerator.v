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
    | (4, 4) => 88
    | (4, 5) => 94
    | (4, 6) => 96
    | (5, 4) => 99
    | (5, 5) => 99
    | (5, 6) => 0
    | (6, 4) => 0
    | (6, 5) => 5
    | (6, 6) => 99
    | _ => 500
    end
    in
    bindGen (freq [
      (weight_1, returnGen 1);
      (100-weight_1, returnGen 0)
    ]) (fun n1 =>
    let weight_2 :=
    match ((stack1 : nat), (stack2 : nat)) with 
    | (4, 4) => 98
    | (4, 5) => 16
    | (4, 6) => 100
    | (5, 4) => 4
    | (5, 5) => 100
    | (5, 6) => 0
    | (6, 4) => 99
    | (6, 5) => 2
    | (6, 6) => 100
    | _ => 500
    end
    in
    bindGen (freq [
      (weight_2, returnGen 2);
      (100-weight_2, returnGen 0)
    ]) (fun n2 =>
    let weight_4 :=
    match ((stack1 : nat), (stack2 : nat)) with 
    | (4, 4) => 95
    | (4, 5) => 99
    | (4, 6) => 8
    | (5, 4) => 0
    | (5, 5) => 4
    | (5, 6) => 90
    | (6, 4) => 96
    | (6, 5) => 4
    | (6, 6) => 9
    | _ => 500
    end
    in
    bindGen (freq [
      (weight_4, returnGen 4);
      (100-weight_4, returnGen 0)
    ]) (fun n4 =>
    let weight_8 :=
    match ((stack1 : nat), (stack2 : nat)) with 
    | (4, 4) => 94
    | (4, 5) => 0
    | (4, 6) => 2
    | (5, 4) => 95
    | (5, 5) => 94
    | (5, 6) => 97
    | (6, 4) => 2
    | (6, 5) => 0
    | (6, 6) => 9
    | _ => 500
    end
    in
    bindGen (freq [
      (weight_8, returnGen 8);
      (100-weight_8, returnGen 0)
    ]) (fun n8 =>
    let weight_16 :=
    match ((stack1 : nat), (stack2 : nat)) with 
    | (4, 4) => 14
    | (4, 5) => 95
    | (4, 6) => 2
    | (5, 4) => 7
    | (5, 5) => 1
    | (5, 6) => 93
    | (6, 4) => 96
    | (6, 5) => 96
    | (6, 6) => 5
    | _ => 500
    end
    in
    bindGen (freq [
      (weight_16, returnGen 16);
      (100-weight_16, returnGen 0)
    ]) (fun n16 =>
    let weight_32 :=
    match ((stack1 : nat), (stack2 : nat)) with 
    | (4, 4) => 99
    | (4, 5) => 0
    | (4, 6) => 23
    | (5, 4) => 94
    | (5, 5) => 0
    | (5, 6) => 4
    | (6, 4) => 1
    | (6, 5) => 100
    | (6, 6) => 89
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
    | (4, 4) => 14
    | (4, 5) => 50
    | (4, 6) => 34
    | (5, 4) => 3
    | (5, 5) => 50
    | (5, 6) => 95
    | (6, 4) => 100
    | (6, 5) => 50
    | (6, 6) => 100
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
          | (1, (0, 3)) => 58
          | (1, (4, 3)) => 37
          | (1, (5, 3)) => 96
          | (1, (6, 3)) => 96
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Typ_ctor Ctor_TBool Ctor_TBool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (0, 3)) => 47
          | (1, (4, 3)) => 89
          | (1, (5, 3)) => 88
          | (1, (6, 3)) => 98
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Typ_ctor Ctor_TFun Ctor_TBool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (0, 3)) => 50
          | (1, (4, 3)) => 93
          | (1, (5, 3)) => 97
          | (1, (6, 3)) => 62
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Typ_ctor Ctor_TBool Ctor_TFun)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (0, 3)) => 45
          | (1, (4, 3)) => 96
          | (1, (5, 3)) => 98
          | (1, (6, 3)) => 97
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
      | (0, (4, 4)) => 59
      | (0, (4, 5)) => 95
      | (0, (4, 6)) => 14
      | (0, (5, 4)) => 7
      | (0, (5, 5)) => 99
      | (0, (5, 6)) => 81
      | (0, (6, 4)) => 1
      | (0, (6, 5)) => 98
      | (0, (6, 6)) => 94
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_1, returnGen 1);
        (100-weight_1, returnGen 0)
      ]) (fun n1 =>
      let weight_2 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (4, 4)) => 10
      | (0, (4, 5)) => 29
      | (0, (4, 6)) => 55
      | (0, (5, 4)) => 1
      | (0, (5, 5)) => 99
      | (0, (5, 6)) => 100
      | (0, (6, 4)) => 0
      | (0, (6, 5)) => 1
      | (0, (6, 6)) => 12
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_2, returnGen 2);
        (100-weight_2, returnGen 0)
      ]) (fun n2 =>
      let weight_4 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (4, 4)) => 18
      | (0, (4, 5)) => 96
      | (0, (4, 6)) => 2
      | (0, (5, 4)) => 56
      | (0, (5, 5)) => 91
      | (0, (5, 6)) => 21
      | (0, (6, 4)) => 67
      | (0, (6, 5)) => 99
      | (0, (6, 6)) => 74
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_4, returnGen 4);
        (100-weight_4, returnGen 0)
      ]) (fun n4 =>
      let weight_8 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (4, 4)) => 95
      | (0, (4, 5)) => 0
      | (0, (4, 6)) => 98
      | (0, (5, 4)) => 94
      | (0, (5, 5)) => 97
      | (0, (5, 6)) => 81
      | (0, (6, 4)) => 99
      | (0, (6, 5)) => 4
      | (0, (6, 6)) => 1
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_8, returnGen 8);
        (100-weight_8, returnGen 0)
      ]) (fun n8 =>
      let weight_16 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (4, 4)) => 95
      | (0, (4, 5)) => 4
      | (0, (4, 6)) => 34
      | (0, (5, 4)) => 1
      | (0, (5, 5)) => 87
      | (0, (5, 6)) => 90
      | (0, (6, 4)) => 97
      | (0, (6, 5)) => 6
      | (0, (6, 6)) => 8
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_16, returnGen 16);
        (100-weight_16, returnGen 0)
      ]) (fun n16 =>
      let weight_32 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (0, (4, 4)) => 4
      | (0, (4, 5)) => 99
      | (0, (4, 6)) => 92
      | (0, (5, 4)) => 0
      | (0, (5, 5)) => 5
      | (0, (5, 6)) => 92
      | (0, (6, 4)) => 5
      | (0, (6, 5)) => 13
      | (0, (6, 6)) => 92
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
      | (0, (4, 4)) => 65
      | (0, (4, 5)) => 50
      | (0, (4, 6)) => 48
      | (0, (5, 4)) => 1
      | (0, (5, 5)) => 50
      | (0, (5, 6)) => 2
      | (0, (6, 4)) => 42
      | (0, (6, 5)) => 50
      | (0, (6, 6)) => 7
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
          | (0, (4, 4)) => 85
          | (0, (4, 5)) => 81
          | (0, (4, 6)) => 10
          | (0, (5, 4)) => 45
          | (0, (5, 5)) => 2
          | (0, (5, 6)) => 0
          | (0, (6, 4)) => 33
          | (0, (6, 5)) => 42
          | (0, (6, 6)) => 0
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_leaf_ctor Ctor_TBool Ctor_leaf_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (4, 4)) => 66
          | (0, (4, 5)) => 87
          | (0, (4, 6)) => 92
          | (0, (5, 4)) => 97
          | (0, (5, 5)) => 97
          | (0, (5, 6)) => 97
          | (0, (6, 4)) => 96
          | (0, (6, 5)) => 94
          | (0, (6, 6)) => 98
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_leaf_ctor Ctor_TFun Ctor_leaf_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (4, 4)) => 5
          | (0, (4, 5)) => 44
          | (0, (4, 6)) => 7
          | (0, (5, 4)) => 13
          | (0, (5, 5)) => 5
          | (0, (5, 6)) => 1
          | (0, (6, 4)) => 0
          | (0, (6, 5)) => 3
          | (0, (6, 6)) => 12
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_leaf_ctor Ctor_TBool Ctor_leaf_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (4, 4)) => 37
          | (0, (4, 5)) => 1
          | (0, (4, 6)) => 59
          | (0, (5, 4)) => 83
          | (0, (5, 5)) => 60
          | (0, (5, 6)) => 32
          | (0, (6, 4)) => 2
          | (0, (6, 5)) => 97
          | (0, (6, 6)) => 4
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
          | (0, (4, 4)) => 89
          | (0, (4, 5)) => 94
          | (0, (4, 6)) => 95
          | (0, (5, 4)) => 97
          | (0, (5, 5)) => 97
          | (0, (5, 6)) => 98
          | (0, (6, 4)) => 97
          | (0, (6, 5)) => 97
          | (0, (6, 6)) => 97
          | _ => 500
          end,
          returnGen (MkExpr_leaf_ctor_Expr_leaf_ctor Ctor_leaf_Var Ctor_leaf_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (4, 4)) => 0
          | (0, (4, 5)) => 0
          | (0, (4, 6)) => 0
          | (0, (5, 4)) => 0
          | (0, (5, 5)) => 0
          | (0, (5, 6)) => 0
          | (0, (6, 4)) => 0
          | (0, (6, 5)) => 0
          | (0, (6, 6)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_leaf_ctor_Expr_leaf_ctor Ctor_leaf_Bool Ctor_leaf_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (4, 4)) => 84
          | (0, (4, 5)) => 70
          | (0, (4, 6)) => 29
          | (0, (5, 4)) => 47
          | (0, (5, 5)) => 92
          | (0, (5, 6)) => 52
          | (0, (6, 4)) => 46
          | (0, (6, 5)) => 88
          | (0, (6, 6)) => 73
          | _ => 500
          end,
          returnGen (MkExpr_leaf_ctor_Expr_leaf_ctor Ctor_leaf_Var Ctor_leaf_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (0, (4, 4)) => 0
          | (0, (4, 5)) => 0
          | (0, (4, 6)) => 0
          | (0, (5, 4)) => 0
          | (0, (5, 5)) => 0
          | (0, (5, 6)) => 0
          | (0, (6, 4)) => 0
          | (0, (6, 5)) => 0
          | (0, (6, 6)) => 0
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
      | (1, (4, 4)) => 21
      | (1, (4, 5)) => 88
      | (1, (4, 6)) => 87
      | (1, (5, 4)) => 6
      | (1, (5, 5)) => 95
      | (1, (5, 6)) => 93
      | (1, (6, 4)) => 90
      | (1, (6, 5)) => 86
      | (1, (6, 6)) => 73
      | (2, (4, 4)) => 55
      | (2, (4, 5)) => 55
      | (2, (4, 6)) => 44
      | (2, (5, 4)) => 27
      | (2, (5, 5)) => 96
      | (2, (5, 6)) => 14
      | (2, (6, 4)) => 41
      | (2, (6, 5)) => 90
      | (2, (6, 6)) => 51
      | (3, (0, 4)) => 51
      | (3, (0, 5)) => 29
      | (3, (0, 6)) => 37
      | (4, (0, 0)) => 60
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_1, returnGen 1);
        (100-weight_1, returnGen 0)
      ]) (fun n1 =>
      let weight_2 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (4, 4)) => 23
      | (1, (4, 5)) => 92
      | (1, (4, 6)) => 79
      | (1, (5, 4)) => 91
      | (1, (5, 5)) => 45
      | (1, (5, 6)) => 0
      | (1, (6, 4)) => 18
      | (1, (6, 5)) => 2
      | (1, (6, 6)) => 2
      | (2, (4, 4)) => 37
      | (2, (4, 5)) => 52
      | (2, (4, 6)) => 71
      | (2, (5, 4)) => 35
      | (2, (5, 5)) => 81
      | (2, (5, 6)) => 60
      | (2, (6, 4)) => 17
      | (2, (6, 5)) => 98
      | (2, (6, 6)) => 39
      | (3, (0, 4)) => 52
      | (3, (0, 5)) => 32
      | (3, (0, 6)) => 78
      | (4, (0, 0)) => 48
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_2, returnGen 2);
        (100-weight_2, returnGen 0)
      ]) (fun n2 =>
      let weight_4 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (4, 4)) => 29
      | (1, (4, 5)) => 12
      | (1, (4, 6)) => 37
      | (1, (5, 4)) => 48
      | (1, (5, 5)) => 57
      | (1, (5, 6)) => 98
      | (1, (6, 4)) => 3
      | (1, (6, 5)) => 51
      | (1, (6, 6)) => 91
      | (2, (4, 4)) => 55
      | (2, (4, 5)) => 48
      | (2, (4, 6)) => 62
      | (2, (5, 4)) => 26
      | (2, (5, 5)) => 18
      | (2, (5, 6)) => 67
      | (2, (6, 4)) => 84
      | (2, (6, 5)) => 72
      | (2, (6, 6)) => 43
      | (3, (0, 4)) => 41
      | (3, (0, 5)) => 70
      | (3, (0, 6)) => 36
      | (4, (0, 0)) => 57
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_4, returnGen 4);
        (100-weight_4, returnGen 0)
      ]) (fun n4 =>
      let weight_8 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (4, 4)) => 41
      | (1, (4, 5)) => 75
      | (1, (4, 6)) => 58
      | (1, (5, 4)) => 8
      | (1, (5, 5)) => 4
      | (1, (5, 6)) => 96
      | (1, (6, 4)) => 94
      | (1, (6, 5)) => 78
      | (1, (6, 6)) => 100
      | (2, (4, 4)) => 49
      | (2, (4, 5)) => 35
      | (2, (4, 6)) => 65
      | (2, (5, 4)) => 76
      | (2, (5, 5)) => 89
      | (2, (5, 6)) => 89
      | (2, (6, 4)) => 84
      | (2, (6, 5)) => 83
      | (2, (6, 6)) => 77
      | (3, (0, 4)) => 50
      | (3, (0, 5)) => 36
      | (3, (0, 6)) => 86
      | (4, (0, 0)) => 53
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_8, returnGen 8);
        (100-weight_8, returnGen 0)
      ]) (fun n8 =>
      let weight_16 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (4, 4)) => 54
      | (1, (4, 5)) => 19
      | (1, (4, 6)) => 68
      | (1, (5, 4)) => 5
      | (1, (5, 5)) => 46
      | (1, (5, 6)) => 3
      | (1, (6, 4)) => 54
      | (1, (6, 5)) => 4
      | (1, (6, 6)) => 99
      | (2, (4, 4)) => 50
      | (2, (4, 5)) => 55
      | (2, (4, 6)) => 41
      | (2, (5, 4)) => 45
      | (2, (5, 5)) => 31
      | (2, (5, 6)) => 25
      | (2, (6, 4)) => 27
      | (2, (6, 5)) => 54
      | (2, (6, 6)) => 95
      | (3, (0, 4)) => 43
      | (3, (0, 5)) => 40
      | (3, (0, 6)) => 89
      | (4, (0, 0)) => 48
      | _ => 500
      end
      in
      bindGen (freq [
        (weight_16, returnGen 16);
        (100-weight_16, returnGen 0)
      ]) (fun n16 =>
      let weight_32 :=
      match (size, ((stack1 : nat), (stack2 : nat))) with 
      | (1, (4, 4)) => 15
      | (1, (4, 5)) => 85
      | (1, (4, 6)) => 50
      | (1, (5, 4)) => 48
      | (1, (5, 5)) => 35
      | (1, (5, 6)) => 80
      | (1, (6, 4)) => 96
      | (1, (6, 5)) => 6
      | (1, (6, 6)) => 2
      | (2, (4, 4)) => 50
      | (2, (4, 5)) => 49
      | (2, (4, 6)) => 55
      | (2, (5, 4)) => 30
      | (2, (5, 5)) => 37
      | (2, (5, 6)) => 70
      | (2, (6, 4)) => 69
      | (2, (6, 5)) => 78
      | (2, (6, 6)) => 55
      | (3, (0, 4)) => 62
      | (3, (0, 5)) => 71
      | (3, (0, 6)) => 77
      | (4, (0, 0)) => 46
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
      | (1, (4, 4)) => 21
      | (1, (4, 5)) => 50
      | (1, (4, 6)) => 79
      | (1, (5, 4)) => 9
      | (1, (5, 5)) => 50
      | (1, (5, 6)) => 4
      | (1, (6, 4)) => 13
      | (1, (6, 5)) => 50
      | (1, (6, 6)) => 2
      | (2, (4, 4)) => 50
      | (2, (4, 5)) => 50
      | (2, (4, 6)) => 42
      | (2, (5, 4)) => 38
      | (2, (5, 5)) => 50
      | (2, (5, 6)) => 81
      | (2, (6, 4)) => 55
      | (2, (6, 5)) => 50
      | (2, (6, 6)) => 94
      | (3, (0, 4)) => 45
      | (3, (0, 5)) => 50
      | (3, (0, 6)) => 42
      | (4, (0, 0)) => 51
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
          | (1, (4, 4)) => 72
          | (1, (4, 5)) => 23
          | (1, (4, 6)) => 61
          | (1, (5, 4)) => 59
          | (1, (5, 5)) => 3
          | (1, (5, 6)) => 3
          | (1, (6, 4)) => 12
          | (1, (6, 5)) => 0
          | (1, (6, 6)) => 0
          | (2, (4, 4)) => 49
          | (2, (4, 5)) => 45
          | (2, (4, 6)) => 50
          | (2, (5, 4)) => 15
          | (2, (5, 5)) => 1
          | (2, (5, 6)) => 1
          | (2, (6, 4)) => 71
          | (2, (6, 5)) => 10
          | (2, (6, 6)) => 5
          | (3, (0, 4)) => 40
          | (3, (0, 5)) => 7
          | (3, (0, 6)) => 4
          | (4, (0, 0)) => 25
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TBool Ctor_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 24
          | (1, (4, 5)) => 65
          | (1, (4, 6)) => 68
          | (1, (5, 4)) => 60
          | (1, (5, 5)) => 78
          | (1, (5, 6)) => 77
          | (1, (6, 4)) => 81
          | (1, (6, 5)) => 94
          | (1, (6, 6)) => 1
          | (2, (4, 4)) => 50
          | (2, (4, 5)) => 42
          | (2, (4, 6)) => 53
          | (2, (5, 4)) => 14
          | (2, (5, 5)) => 19
          | (2, (5, 6)) => 2
          | (2, (6, 4)) => 47
          | (2, (6, 5)) => 5
          | (2, (6, 6)) => 2
          | (3, (0, 4)) => 48
          | (3, (0, 5)) => 34
          | (3, (0, 6)) => 10
          | (4, (0, 0)) => 25
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TFun Ctor_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 13
          | (1, (4, 5)) => 22
          | (1, (4, 6)) => 55
          | (1, (5, 4)) => 1
          | (1, (5, 5)) => 0
          | (1, (5, 6)) => 6
          | (1, (6, 4)) => 30
          | (1, (6, 5)) => 0
          | (1, (6, 6)) => 0
          | (2, (4, 4)) => 48
          | (2, (4, 5)) => 47
          | (2, (4, 6)) => 50
          | (2, (5, 4)) => 13
          | (2, (5, 5)) => 1
          | (2, (5, 6)) => 3
          | (2, (6, 4)) => 6
          | (2, (6, 5)) => 2
          | (2, (6, 6)) => 0
          | (3, (0, 4)) => 30
          | (3, (0, 5)) => 2
          | (3, (0, 6)) => 4
          | (4, (0, 0)) => 11
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TBool Ctor_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 47
          | (1, (4, 5)) => 59
          | (1, (4, 6)) => 32
          | (1, (5, 4)) => 5
          | (1, (5, 5)) => 3
          | (1, (5, 6)) => 0
          | (1, (6, 4)) => 6
          | (1, (6, 5)) => 11
          | (1, (6, 6)) => 0
          | (2, (4, 4)) => 44
          | (2, (4, 5)) => 48
          | (2, (4, 6)) => 50
          | (2, (5, 4)) => 16
          | (2, (5, 5)) => 5
          | (2, (5, 6)) => 15
          | (2, (6, 4)) => 18
          | (2, (6, 5)) => 1
          | (2, (6, 6)) => 3
          | (3, (0, 4)) => 33
          | (3, (0, 5)) => 3
          | (3, (0, 6)) => 4
          | (4, (0, 0)) => 13
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TFun Ctor_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 46
          | (1, (4, 5)) => 78
          | (1, (4, 6)) => 45
          | (1, (5, 4)) => 84
          | (1, (5, 5)) => 57
          | (1, (5, 6)) => 3
          | (1, (6, 4)) => 26
          | (1, (6, 5)) => 3
          | (1, (6, 6)) => 77
          | (2, (4, 4)) => 55
          | (2, (4, 5)) => 58
          | (2, (4, 6)) => 55
          | (2, (5, 4)) => 70
          | (2, (5, 5)) => 23
          | (2, (5, 6)) => 29
          | (2, (6, 4)) => 67
          | (2, (6, 5)) => 1
          | (2, (6, 6)) => 27
          | (3, (0, 4)) => 57
          | (3, (0, 5)) => 60
          | (3, (0, 6)) => 57
          | (4, (0, 0)) => 64
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TBool Ctor_Abs)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 43
          | (1, (4, 5)) => 67
          | (1, (4, 6)) => 38
          | (1, (5, 4)) => 88
          | (1, (5, 5)) => 98
          | (1, (5, 6)) => 96
          | (1, (6, 4)) => 90
          | (1, (6, 5)) => 96
          | (1, (6, 6)) => 98
          | (2, (4, 4)) => 49
          | (2, (4, 5)) => 51
          | (2, (4, 6)) => 50
          | (2, (5, 4)) => 68
          | (2, (5, 5)) => 65
          | (2, (5, 6)) => 74
          | (2, (6, 4)) => 77
          | (2, (6, 5)) => 92
          | (2, (6, 6)) => 35
          | (3, (0, 4)) => 69
          | (3, (0, 5)) => 91
          | (3, (0, 6)) => 74
          | (4, (0, 0)) => 76
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TFun Ctor_Abs)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 61
          | (1, (4, 5)) => 22
          | (1, (4, 6)) => 45
          | (1, (5, 4)) => 6
          | (1, (5, 5)) => 3
          | (1, (5, 6)) => 0
          | (1, (6, 4)) => 34
          | (1, (6, 5)) => 0
          | (1, (6, 6)) => 2
          | (2, (4, 4)) => 52
          | (2, (4, 5)) => 47
          | (2, (4, 6)) => 50
          | (2, (5, 4)) => 70
          | (2, (5, 5)) => 89
          | (2, (5, 6)) => 85
          | (2, (6, 4)) => 62
          | (2, (6, 5)) => 83
          | (2, (6, 6)) => 63
          | (3, (0, 4)) => 58
          | (3, (0, 5)) => 54
          | (3, (0, 6)) => 14
          | (4, (0, 0)) => 55
          | _ => 500
          end,
          returnGen (MkTyp_ctor_Expr_ctor Ctor_TBool Ctor_App)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 76
          | (1, (4, 5)) => 38
          | (1, (4, 6)) => 55
          | (1, (5, 4)) => 78
          | (1, (5, 5)) => 13
          | (1, (5, 6)) => 98
          | (1, (6, 4)) => 61
          | (1, (6, 5)) => 97
          | (1, (6, 6)) => 95
          | (2, (4, 4)) => 53
          | (2, (4, 5)) => 58
          | (2, (4, 6)) => 43
          | (2, (5, 4)) => 83
          | (2, (5, 5)) => 94
          | (2, (5, 6)) => 94
          | (2, (6, 4)) => 32
          | (2, (6, 5)) => 92
          | (2, (6, 6)) => 96
          | (3, (0, 4)) => 53
          | (3, (0, 5)) => 78
          | (3, (0, 6)) => 94
          | (4, (0, 0)) => 80
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
          | (1, (4, 4)) => 74
          | (1, (4, 5)) => 67
          | (1, (4, 6)) => 77
          | (1, (5, 4)) => 72
          | (1, (5, 5)) => 1
          | (1, (5, 6)) => 5
          | (1, (6, 4)) => 34
          | (1, (6, 5)) => 10
          | (1, (6, 6)) => 1
          | (2, (4, 4)) => 51
          | (2, (4, 5)) => 56
          | (2, (4, 6)) => 57
          | (2, (5, 4)) => 49
          | (2, (5, 5)) => 0
          | (2, (5, 6)) => 0
          | (2, (6, 4)) => 45
          | (2, (6, 5)) => 0
          | (2, (6, 6)) => 1
          | (3, (0, 4)) => 57
          | (3, (0, 5)) => 0
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Var Ctor_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 5
          | (1, (4, 5)) => 7
          | (1, (4, 6)) => 7
          | (1, (5, 4)) => 0
          | (1, (5, 5)) => 0
          | (1, (5, 6)) => 0
          | (1, (6, 4)) => 0
          | (1, (6, 5)) => 0
          | (1, (6, 6)) => 0
          | (2, (4, 4)) => 41
          | (2, (4, 5)) => 41
          | (2, (4, 6)) => 45
          | (2, (5, 4)) => 4
          | (2, (5, 5)) => 0
          | (2, (5, 6)) => 0
          | (2, (6, 4)) => 4
          | (2, (6, 5)) => 0
          | (2, (6, 6)) => 0
          | (3, (0, 4)) => 25
          | (3, (0, 5)) => 0
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Bool Ctor_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 57
          | (1, (4, 5)) => 71
          | (1, (4, 6)) => 70
          | (1, (5, 4)) => 88
          | (1, (5, 5)) => 91
          | (1, (5, 6)) => 93
          | (1, (6, 4)) => 82
          | (1, (6, 5)) => 1
          | (1, (6, 6)) => 93
          | (2, (4, 4)) => 54
          | (2, (4, 5)) => 63
          | (2, (4, 6)) => 48
          | (2, (5, 4)) => 68
          | (2, (5, 5)) => 0
          | (2, (5, 6)) => 0
          | (2, (6, 4)) => 55
          | (2, (6, 5)) => 1
          | (2, (6, 6)) => 0
          | (3, (0, 4)) => 62
          | (3, (0, 5)) => 0
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Abs Ctor_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 68
          | (1, (4, 5)) => 49
          | (1, (4, 6)) => 35
          | (1, (5, 4)) => 77
          | (1, (5, 5)) => 96
          | (1, (5, 6)) => 93
          | (1, (6, 4)) => 22
          | (1, (6, 5)) => 3
          | (1, (6, 6)) => 3
          | (2, (4, 4)) => 48
          | (2, (4, 5)) => 49
          | (2, (4, 6)) => 51
          | (2, (5, 4)) => 69
          | (2, (5, 5)) => 0
          | (2, (5, 6)) => 0
          | (2, (6, 4)) => 40
          | (2, (6, 5)) => 1
          | (2, (6, 6)) => 1
          | (3, (0, 4)) => 62
          | (3, (0, 5)) => 0
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_App Ctor_Var)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 57
          | (1, (4, 5)) => 45
          | (1, (4, 6)) => 61
          | (1, (5, 4)) => 32
          | (1, (5, 5)) => 0
          | (1, (5, 6)) => 10
          | (1, (6, 4)) => 2
          | (1, (6, 5)) => 0
          | (1, (6, 6)) => 1
          | (2, (4, 4)) => 53
          | (2, (4, 5)) => 56
          | (2, (4, 6)) => 51
          | (2, (5, 4)) => 22
          | (2, (5, 5)) => 0
          | (2, (5, 6)) => 0
          | (2, (6, 4)) => 43
          | (2, (6, 5)) => 0
          | (2, (6, 6)) => 0
          | (3, (0, 4)) => 48
          | (3, (0, 5)) => 0
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Var Ctor_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 5
          | (1, (4, 5)) => 7
          | (1, (4, 6)) => 7
          | (1, (5, 4)) => 0
          | (1, (5, 5)) => 0
          | (1, (5, 6)) => 0
          | (1, (6, 4)) => 0
          | (1, (6, 5)) => 0
          | (1, (6, 6)) => 0
          | (2, (4, 4)) => 41
          | (2, (4, 5)) => 41
          | (2, (4, 6)) => 45
          | (2, (5, 4)) => 4
          | (2, (5, 5)) => 0
          | (2, (5, 6)) => 0
          | (2, (6, 4)) => 4
          | (2, (6, 5)) => 0
          | (2, (6, 6)) => 0
          | (3, (0, 4)) => 25
          | (3, (0, 5)) => 0
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Bool Ctor_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 53
          | (1, (4, 5)) => 48
          | (1, (4, 6)) => 57
          | (1, (5, 4)) => 26
          | (1, (5, 5)) => 6
          | (1, (5, 6)) => 45
          | (1, (6, 4)) => 62
          | (1, (6, 5)) => 1
          | (1, (6, 6)) => 1
          | (2, (4, 4)) => 54
          | (2, (4, 5)) => 50
          | (2, (4, 6)) => 50
          | (2, (5, 4)) => 45
          | (2, (5, 5)) => 0
          | (2, (5, 6)) => 0
          | (2, (6, 4)) => 69
          | (2, (6, 5)) => 0
          | (2, (6, 6)) => 1
          | (3, (0, 4)) => 58
          | (3, (0, 5)) => 0
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Abs Ctor_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 33
          | (1, (4, 5)) => 61
          | (1, (4, 6)) => 56
          | (1, (5, 4)) => 10
          | (1, (5, 5)) => 1
          | (1, (5, 6)) => 0
          | (1, (6, 4)) => 2
          | (1, (6, 5)) => 0
          | (1, (6, 6)) => 1
          | (2, (4, 4)) => 54
          | (2, (4, 5)) => 48
          | (2, (4, 6)) => 51
          | (2, (5, 4)) => 31
          | (2, (5, 5)) => 0
          | (2, (5, 6)) => 0
          | (2, (6, 4)) => 38
          | (2, (6, 5)) => 0
          | (2, (6, 6)) => 0
          | (3, (0, 4)) => 49
          | (3, (0, 5)) => 0
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_App Ctor_Bool)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 65
          | (1, (4, 5)) => 72
          | (1, (4, 6)) => 66
          | (1, (5, 4)) => 83
          | (1, (5, 5)) => 3
          | (1, (5, 6)) => 91
          | (1, (6, 4)) => 60
          | (1, (6, 5)) => 91
          | (1, (6, 6)) => 95
          | (2, (4, 4)) => 56
          | (2, (4, 5)) => 47
          | (2, (4, 6)) => 52
          | (2, (5, 4)) => 69
          | (2, (5, 5)) => 1
          | (2, (5, 6)) => 1
          | (2, (6, 4)) => 70
          | (2, (6, 5)) => 0
          | (2, (6, 6)) => 0
          | (3, (0, 4)) => 63
          | (3, (0, 5)) => 0
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Var Ctor_Abs)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 5
          | (1, (4, 5)) => 7
          | (1, (4, 6)) => 7
          | (1, (5, 4)) => 0
          | (1, (5, 5)) => 0
          | (1, (5, 6)) => 0
          | (1, (6, 4)) => 0
          | (1, (6, 5)) => 0
          | (1, (6, 6)) => 0
          | (2, (4, 4)) => 41
          | (2, (4, 5)) => 41
          | (2, (4, 6)) => 45
          | (2, (5, 4)) => 4
          | (2, (5, 5)) => 0
          | (2, (5, 6)) => 0
          | (2, (6, 4)) => 4
          | (2, (6, 5)) => 0
          | (2, (6, 6)) => 0
          | (3, (0, 4)) => 25
          | (3, (0, 5)) => 0
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Bool Ctor_Abs)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 74
          | (1, (4, 5)) => 66
          | (1, (4, 6)) => 56
          | (1, (5, 4)) => 80
          | (1, (5, 5)) => 99
          | (1, (5, 6)) => 98
          | (1, (6, 4)) => 92
          | (1, (6, 5)) => 88
          | (1, (6, 6)) => 98
          | (2, (4, 4)) => 56
          | (2, (4, 5)) => 53
          | (2, (4, 6)) => 52
          | (2, (5, 4)) => 66
          | (2, (5, 5)) => 13
          | (2, (5, 6)) => 0
          | (2, (6, 4)) => 78
          | (2, (6, 5)) => 1
          | (2, (6, 6)) => 26
          | (3, (0, 4)) => 71
          | (3, (0, 5)) => 0
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Abs Ctor_Abs)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 72
          | (1, (4, 5)) => 75
          | (1, (4, 6)) => 51
          | (1, (5, 4)) => 78
          | (1, (5, 5)) => 84
          | (1, (5, 6)) => 95
          | (1, (6, 4)) => 85
          | (1, (6, 5)) => 98
          | (1, (6, 6)) => 87
          | (2, (4, 4)) => 48
          | (2, (4, 5)) => 56
          | (2, (4, 6)) => 55
          | (2, (5, 4)) => 72
          | (2, (5, 5)) => 81
          | (2, (5, 6)) => 81
          | (2, (6, 4)) => 75
          | (2, (6, 5)) => 6
          | (2, (6, 6)) => 90
          | (3, (0, 4)) => 56
          | (3, (0, 5)) => 0
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_App Ctor_Abs)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 65
          | (1, (4, 5)) => 74
          | (1, (4, 6)) => 66
          | (1, (5, 4)) => 92
          | (1, (5, 5)) => 20
          | (1, (5, 6)) => 44
          | (1, (6, 4)) => 92
          | (1, (6, 5)) => 3
          | (1, (6, 6)) => 30
          | (2, (4, 4)) => 55
          | (2, (4, 5)) => 55
          | (2, (4, 6)) => 49
          | (2, (5, 4)) => 80
          | (2, (5, 5)) => 0
          | (2, (5, 6)) => 0
          | (2, (6, 4)) => 46
          | (2, (6, 5)) => 0
          | (2, (6, 6)) => 0
          | (3, (0, 4)) => 47
          | (3, (0, 5)) => 0
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Var Ctor_App)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 5
          | (1, (4, 5)) => 7
          | (1, (4, 6)) => 7
          | (1, (5, 4)) => 0
          | (1, (5, 5)) => 0
          | (1, (5, 6)) => 0
          | (1, (6, 4)) => 0
          | (1, (6, 5)) => 0
          | (1, (6, 6)) => 0
          | (2, (4, 4)) => 41
          | (2, (4, 5)) => 41
          | (2, (4, 6)) => 45
          | (2, (5, 4)) => 4
          | (2, (5, 5)) => 0
          | (2, (5, 6)) => 0
          | (2, (6, 4)) => 4
          | (2, (6, 5)) => 0
          | (2, (6, 6)) => 0
          | (3, (0, 4)) => 25
          | (3, (0, 5)) => 0
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Bool Ctor_App)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 50
          | (1, (4, 5)) => 58
          | (1, (4, 6)) => 74
          | (1, (5, 4)) => 83
          | (1, (5, 5)) => 97
          | (1, (5, 6)) => 97
          | (1, (6, 4)) => 87
          | (1, (6, 5)) => 97
          | (1, (6, 6)) => 97
          | (2, (4, 4)) => 48
          | (2, (4, 5)) => 50
          | (2, (4, 6)) => 54
          | (2, (5, 4)) => 84
          | (2, (5, 5)) => 94
          | (2, (5, 6)) => 77
          | (2, (6, 4)) => 85
          | (2, (6, 5)) => 16
          | (2, (6, 6)) => 94
          | (3, (0, 4)) => 48
          | (3, (0, 5)) => 4
          | (3, (0, 6)) => 0
          | (4, (0, 0)) => 0
          | _ => 500
          end,
          returnGen (MkExpr_ctor_Expr_ctor Ctor_Abs Ctor_App)
        );
        (
          match (size, ((stack1 : nat), (stack2 : nat))) with 
          | (1, (4, 4)) => 75
          | (1, (4, 5)) => 52
          | (1, (4, 6)) => 66
          | (1, (5, 4)) => 62
          | (1, (5, 5)) => 7
          | (1, (5, 6)) => 88
          | (1, (6, 4)) => 85
          | (1, (6, 5)) => 98
          | (1, (6, 6)) => 91
          | (2, (4, 4)) => 53
          | (2, (4, 5)) => 47
          | (2, (4, 6)) => 49
          | (2, (5, 4)) => 66
          | (2, (5, 5)) => 99
          | (2, (5, 6)) => 99
          | (2, (6, 4)) => 74
          | (2, (6, 5)) => 99
          | (2, (6, 6)) => 99
          | (3, (0, 4)) => 50
          | (3, (0, 5)) => 99
          | (3, (0, 6)) => 99
          | (4, (0, 0)) => 99
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
      0,
      returnGen Ctor_Bool
    );
    (
      0,
      returnGen Ctor_Abs
    );
    (
      97,
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
          
