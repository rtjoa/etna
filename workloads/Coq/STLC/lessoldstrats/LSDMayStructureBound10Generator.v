From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Inductive CtorExpr :=
  | CtorExpr_Var
  | CtorExpr_Bool
  | CtorExpr_Abs
  | CtorExpr_App.

Inductive LeafCtorExpr :=
  | LeafCtorExpr_Var
  | LeafCtorExpr_Bool.

Inductive CtorTyp :=
  | CtorTyp_TBool
  | CtorTyp_TFun.

Inductive LeafCtorTyp :=
  | LeafCtorTyp_TBool.

Inductive TupCtorTypLeafCtorExpr :=
  | MkCtorTypLeafCtorExpr : CtorTyp -> LeafCtorExpr -> TupCtorTypLeafCtorExpr.

Inductive TupLeafCtorTypLeafCtorTyp :=
  | MkLeafCtorTypLeafCtorTyp : LeafCtorTyp -> LeafCtorTyp -> TupLeafCtorTypLeafCtorTyp.

Inductive TupCtorTypCtorExpr :=
  | MkCtorTypCtorExpr : CtorTyp -> CtorExpr -> TupCtorTypCtorExpr.

Inductive TupCtorTypCtorTyp :=
  | MkCtorTypCtorTyp : CtorTyp -> CtorTyp -> TupCtorTypCtorTyp.

Inductive TupCtorExprCtorExpr :=
  | MkCtorExprCtorExpr : CtorExpr -> CtorExpr -> TupCtorExprCtorExpr.

Inductive TupLeafCtorExprLeafCtorExpr :=
  | MkLeafCtorExprLeafCtorExpr : LeafCtorExpr -> LeafCtorExpr -> TupLeafCtorExprLeafCtorExpr.

Definition genLeafTyp (chosen_ctor : LeafCtorTyp) (stack1 : nat) : G (Typ) :=
  match chosen_ctor with
  | LeafCtorTyp_TBool  => 
    (returnGen (TBool ))
  end.

Fixpoint genTyp (size : nat) (chosen_ctor : CtorTyp) (stack1 : nat) : G (Typ) :=
  match size with
  | O  => match chosen_ctor with
    | CtorTyp_TBool  => 
      (returnGen (TBool ))
    | CtorTyp_TFun  => 
      (bindGen 
      (* Frequency2 (single-branch) *) 
      (returnGen (MkLeafCtorTypLeafCtorTyp (LeafCtorTyp_TBool ) (LeafCtorTyp_TBool ))) 
      (fun param_variantis => (let '(MkLeafCtorTypLeafCtorTyp ctor1 ctor2) := param_variantis in

        (bindGen (genLeafTyp ctor1 1) 
        (fun p1 => 
          (bindGen (genLeafTyp ctor2 7) 
          (fun p2 => 
            (returnGen (TFun p1 p2)))))))))
    end
  | S size1 => match chosen_ctor with
    | CtorTyp_TBool  => 
      (returnGen (TBool ))
    | CtorTyp_TFun  => 
      (bindGen 
      (* Frequency3 *) (freq [
        (* 1 *) (match (size, stack1) with
        | (1, 3) => 50
        | (1, 5) => 50
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorTyp (CtorTyp_TBool ) (CtorTyp_TBool )))); 
        (* 2 *) (match (size, stack1) with
        | (1, 3) => 50
        | (1, 5) => 50
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorTyp (CtorTyp_TFun ) (CtorTyp_TBool )))); 
        (* 3 *) (match (size, stack1) with
        | (1, 3) => 50
        | (1, 5) => 50
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorTyp (CtorTyp_TBool ) (CtorTyp_TFun )))); 
        (* 4 *) (match (size, stack1) with
        | (1, 3) => 50
        | (1, 5) => 50
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorTyp (CtorTyp_TFun ) (CtorTyp_TFun ))))]) 
      (fun param_variantis => (let '(MkCtorTypCtorTyp ctor1 ctor2) := param_variantis in

        (bindGen (genTyp size1 ctor1 2) 
        (fun p1 => 
          (bindGen (genTyp size1 ctor2 8) 
          (fun p2 => 
            (returnGen (TFun p1 p2)))))))))
    end
  end.

Definition genLeafExpr (chosen_ctor : LeafCtorExpr) (stack1 : nat) : G (Expr) :=
  match chosen_ctor with
  | LeafCtorExpr_Var  => 
    (bindGen 
    (* GenNat1 *)
    (let _weight_1 := match (stack1) with
    | (10) => 50
    | (4) => 50
    | (9) => 50
    | _ => 500
    end
    in
    bindGen (freq [
      (_weight_1, returnGen 1);
      (100-_weight_1, returnGen 0)
    ]) (fun n1 =>
    (let _weight_2 := match (stack1) with
    | (10) => 50
    | (4) => 50
    | (9) => 50
    | _ => 500
    end
    in
    bindGen (freq [
      (_weight_2, returnGen 2);
      (100-_weight_2, returnGen 0)
    ]) (fun n2 =>
    (let _weight_4 := match (stack1) with
    | (10) => 50
    | (4) => 50
    | (9) => 50
    | _ => 500
    end
    in
    bindGen (freq [
      (_weight_4, returnGen 4);
      (100-_weight_4, returnGen 0)
    ]) (fun n4 =>
      returnGen (n1 + n2 + n4)
    )))))) 
    (fun p1 => 
      (returnGen (Var p1))))
  | LeafCtorExpr_Bool  => 
    (bindGen 
    (* GenBool1 *) (let _weight_true := match (stack1) with
    | (10) => 50
    | (4) => 50
    | (9) => 50
    | _ => 500
    end
    in
    freq [
      (_weight_true, returnGen true);
      (100-_weight_true, returnGen false)
    ]) 
    (fun p1 => 
      (returnGen (Bool p1))))
  end.

Fixpoint genExpr (size : nat) (chosen_ctor : CtorExpr) (stack1 : nat) : G (Expr) :=
  match size with
  | O  => match chosen_ctor with
    | CtorExpr_Var  => 
      (bindGen 
      (* GenNat2 *)
      (let _weight_1 := match (stack1) with
      | (11) => 50
      | (12) => 50
      | (6) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_1, returnGen 1);
        (100-_weight_1, returnGen 0)
      ]) (fun n1 =>
      (let _weight_2 := match (stack1) with
      | (11) => 50
      | (12) => 50
      | (6) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_2, returnGen 2);
        (100-_weight_2, returnGen 0)
      ]) (fun n2 =>
      (let _weight_4 := match (stack1) with
      | (11) => 50
      | (12) => 50
      | (6) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_4, returnGen 4);
        (100-_weight_4, returnGen 0)
      ]) (fun n4 =>
        returnGen (n1 + n2 + n4)
      )))))) 
      (fun p1 => 
        (returnGen (Var p1))))
    | CtorExpr_Bool  => 
      (bindGen 
      (* GenBool2 *) (let _weight_true := match (stack1) with
      | (11) => 50
      | (12) => 50
      | (6) => 50
      | _ => 500
      end
      in
      freq [
        (_weight_true, returnGen true);
        (100-_weight_true, returnGen false)
      ]) 
      (fun p1 => 
        (returnGen (Bool p1))))
    | CtorExpr_Abs  => 
      (bindGen 
      (* Frequency4 *) (freq [
        (* 1 *) (match (stack1) with
        | (11) => 55
        | (12) => 49
        | (6) => 55
        | _ => 500
        end,
        (returnGen (MkCtorTypLeafCtorExpr (CtorTyp_TBool ) (LeafCtorExpr_Var )))); 
        (* 2 *) (match (stack1) with
        | (11) => 55
        | (12) => 49
        | (6) => 55
        | _ => 500
        end,
        (returnGen (MkCtorTypLeafCtorExpr (CtorTyp_TFun ) (LeafCtorExpr_Var )))); 
        (* 3 *) (match (stack1) with
        | (11) => 48
        | (12) => 54
        | (6) => 49
        | _ => 500
        end,
        (returnGen (MkCtorTypLeafCtorExpr (CtorTyp_TBool ) (LeafCtorExpr_Bool )))); 
        (* 4 *) (match (stack1) with
        | (11) => 48
        | (12) => 54
        | (6) => 49
        | _ => 500
        end,
        (returnGen (MkCtorTypLeafCtorExpr (CtorTyp_TFun ) (LeafCtorExpr_Bool ))))]) 
      (fun param_variantis => (let '(MkCtorTypLeafCtorExpr ctor1 ctor2) := param_variantis in

        (bindGen (genTyp 1 ctor1 3) 
        (fun p1 => 
          (bindGen (genLeafExpr ctor2 9) 
          (fun p2 => 
            (returnGen (Abs p1 p2)))))))))
    | CtorExpr_App  => 
      (bindGen 
      (* Frequency5 *) (freq [
        (* 1 *) (match (stack1) with
        | (11) => 90
        | (12) => 90
        | (6) => 90
        | _ => 500
        end,
        (returnGen (MkLeafCtorExprLeafCtorExpr (LeafCtorExpr_Var ) (LeafCtorExpr_Var )))); 
        (* 2 *) (match (stack1) with
        | (11) => 10
        | (12) => 10
        | (6) => 10
        | _ => 500
        end,
        (returnGen (MkLeafCtorExprLeafCtorExpr (LeafCtorExpr_Bool ) (LeafCtorExpr_Var )))); 
        (* 3 *) (match (stack1) with
        | (11) => 90
        | (12) => 90
        | (6) => 90
        | _ => 500
        end,
        (returnGen (MkLeafCtorExprLeafCtorExpr (LeafCtorExpr_Var ) (LeafCtorExpr_Bool )))); 
        (* 4 *) (match (stack1) with
        | (11) => 10
        | (12) => 10
        | (6) => 10
        | _ => 500
        end,
        (returnGen (MkLeafCtorExprLeafCtorExpr (LeafCtorExpr_Bool ) (LeafCtorExpr_Bool ))))]) 
      (fun param_variantis => (let '(MkLeafCtorExprLeafCtorExpr ctor1 ctor2) := param_variantis in

        (bindGen (genLeafExpr ctor1 4) 
        (fun p1 => 
          (bindGen (genLeafExpr ctor2 10) 
          (fun p2 => 
            (returnGen (App p1 p2)))))))))
    end
  | S size1 => match chosen_ctor with
    | CtorExpr_Var  => 
      (bindGen 
      (* GenNat3 *)
      (let _weight_1 := match (size, stack1) with
      | (1, 11) => 50
      | (1, 12) => 50
      | (1, 6) => 50
      | (2, 11) => 50
      | (2, 12) => 50
      | (2, 6) => 50
      | (3, 11) => 50
      | (3, 12) => 50
      | (3, 6) => 50
      | (4, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_1, returnGen 1);
        (100-_weight_1, returnGen 0)
      ]) (fun n1 =>
      (let _weight_2 := match (size, stack1) with
      | (1, 11) => 50
      | (1, 12) => 50
      | (1, 6) => 50
      | (2, 11) => 50
      | (2, 12) => 50
      | (2, 6) => 50
      | (3, 11) => 50
      | (3, 12) => 50
      | (3, 6) => 50
      | (4, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_2, returnGen 2);
        (100-_weight_2, returnGen 0)
      ]) (fun n2 =>
      (let _weight_4 := match (size, stack1) with
      | (1, 11) => 50
      | (1, 12) => 50
      | (1, 6) => 50
      | (2, 11) => 50
      | (2, 12) => 50
      | (2, 6) => 50
      | (3, 11) => 50
      | (3, 12) => 50
      | (3, 6) => 50
      | (4, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_4, returnGen 4);
        (100-_weight_4, returnGen 0)
      ]) (fun n4 =>
        returnGen (n1 + n2 + n4)
      )))))) 
      (fun p1 => 
        (returnGen (Var p1))))
    | CtorExpr_Bool  => 
      (bindGen 
      (* GenBool3 *) (let _weight_true := match (size, stack1) with
      | (1, 11) => 50
      | (1, 12) => 50
      | (1, 6) => 50
      | (2, 11) => 50
      | (2, 12) => 50
      | (2, 6) => 50
      | (3, 11) => 50
      | (3, 12) => 50
      | (3, 6) => 50
      | (4, 0) => 50
      | _ => 500
      end
      in
      freq [
        (_weight_true, returnGen true);
        (100-_weight_true, returnGen false)
      ]) 
      (fun p1 => 
        (returnGen (Bool p1))))
    | CtorExpr_Abs  => 
      (bindGen 
      (* Frequency6 *) (freq [
        (* 1 *) (match (size, stack1) with
        | (1, 11) => 38
        | (1, 12) => 35
        | (1, 6) => 44
        | (2, 11) => 25
        | (2, 12) => 10
        | (2, 6) => 11
        | (3, 11) => 10
        | (3, 12) => 25
        | (3, 6) => 14
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TBool ) (CtorExpr_Var )))); 
        (* 2 *) (match (size, stack1) with
        | (1, 11) => 38
        | (1, 12) => 35
        | (1, 6) => 44
        | (2, 11) => 25
        | (2, 12) => 10
        | (2, 6) => 11
        | (3, 11) => 10
        | (3, 12) => 25
        | (3, 6) => 14
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TFun ) (CtorExpr_Var )))); 
        (* 3 *) (match (size, stack1) with
        | (1, 11) => 45
        | (1, 12) => 50
        | (1, 6) => 56
        | (2, 11) => 31
        | (2, 12) => 11
        | (2, 6) => 10
        | (3, 11) => 10
        | (3, 12) => 24
        | (3, 6) => 15
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TBool ) (CtorExpr_Bool )))); 
        (* 4 *) (match (size, stack1) with
        | (1, 11) => 45
        | (1, 12) => 50
        | (1, 6) => 56
        | (2, 11) => 31
        | (2, 12) => 11
        | (2, 6) => 10
        | (3, 11) => 10
        | (3, 12) => 24
        | (3, 6) => 15
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TFun ) (CtorExpr_Bool )))); 
        (* 5 *) (match (size, stack1) with
        | (1, 11) => 71
        | (1, 12) => 76
        | (1, 6) => 73
        | (2, 11) => 58
        | (2, 12) => 40
        | (2, 6) => 47
        | (3, 11) => 23
        | (3, 12) => 51
        | (3, 6) => 58
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TBool ) (CtorExpr_Abs )))); 
        (* 6 *) (match (size, stack1) with
        | (1, 11) => 71
        | (1, 12) => 76
        | (1, 6) => 73
        | (2, 11) => 58
        | (2, 12) => 40
        | (2, 6) => 47
        | (3, 11) => 23
        | (3, 12) => 51
        | (3, 6) => 58
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TFun ) (CtorExpr_Abs )))); 
        (* 7 *) (match (size, stack1) with
        | (1, 11) => 38
        | (1, 12) => 23
        | (1, 6) => 17
        | (2, 11) => 72
        | (2, 12) => 87
        | (2, 6) => 86
        | (3, 11) => 90
        | (3, 12) => 77
        | (3, 6) => 81
        | (4, 0) => 90
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TBool ) (CtorExpr_App )))); 
        (* 8 *) (match (size, stack1) with
        | (1, 11) => 38
        | (1, 12) => 23
        | (1, 6) => 17
        | (2, 11) => 72
        | (2, 12) => 87
        | (2, 6) => 86
        | (3, 11) => 90
        | (3, 12) => 77
        | (3, 6) => 81
        | (4, 0) => 90
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TFun ) (CtorExpr_App ))))]) 
      (fun param_variantis => (let '(MkCtorTypCtorExpr ctor1 ctor2) := param_variantis in

        (bindGen (genTyp 1 ctor1 5) 
        (fun p1 => 
          (bindGen (genExpr size1 ctor2 11) 
          (fun p2 => 
            (returnGen (Abs p1 p2)))))))))
    | CtorExpr_App  => 
      (bindGen 
      (* Frequency7 *) (freq [
        (* 1 *) (match (size, stack1) with
        | (1, 11) => 58
        | (1, 12) => 79
        | (1, 6) => 72
        | (2, 11) => 23
        | (2, 12) => 11
        | (2, 6) => 18
        | (3, 11) => 10
        | (3, 12) => 43
        | (3, 6) => 29
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Var ) (CtorExpr_Var )))); 
        (* 2 *) (match (size, stack1) with
        | (1, 11) => 10
        | (1, 12) => 10
        | (1, 6) => 10
        | (2, 11) => 10
        | (2, 12) => 10
        | (2, 6) => 10
        | (3, 11) => 10
        | (3, 12) => 43
        | (3, 6) => 29
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Bool ) (CtorExpr_Var )))); 
        (* 3 *) (match (size, stack1) with
        | (1, 11) => 84
        | (1, 12) => 76
        | (1, 6) => 89
        | (2, 11) => 45
        | (2, 12) => 52
        | (2, 6) => 45
        | (3, 11) => 10
        | (3, 12) => 43
        | (3, 6) => 29
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Abs ) (CtorExpr_Var )))); 
        (* 4 *) (match (size, stack1) with
        | (1, 11) => 54
        | (1, 12) => 63
        | (1, 6) => 54
        | (2, 11) => 69
        | (2, 12) => 64
        | (2, 6) => 76
        | (3, 11) => 23
        | (3, 12) => 43
        | (3, 6) => 29
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_App ) (CtorExpr_Var )))); 
        (* 5 *) (match (size, stack1) with
        | (1, 11) => 78
        | (1, 12) => 82
        | (1, 6) => 86
        | (2, 11) => 27
        | (2, 12) => 18
        | (2, 6) => 12
        | (3, 11) => 10
        | (3, 12) => 43
        | (3, 6) => 29
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Var ) (CtorExpr_Bool )))); 
        (* 6 *) (match (size, stack1) with
        | (1, 11) => 10
        | (1, 12) => 10
        | (1, 6) => 10
        | (2, 11) => 10
        | (2, 12) => 10
        | (2, 6) => 10
        | (3, 11) => 10
        | (3, 12) => 43
        | (3, 6) => 29
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Bool ) (CtorExpr_Bool )))); 
        (* 7 *) (match (size, stack1) with
        | (1, 11) => 90
        | (1, 12) => 86
        | (1, 6) => 89
        | (2, 11) => 54
        | (2, 12) => 51
        | (2, 6) => 55
        | (3, 11) => 10
        | (3, 12) => 72
        | (3, 6) => 85
        | (4, 0) => 90
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Abs ) (CtorExpr_Bool )))); 
        (* 8 *) (match (size, stack1) with
        | (1, 11) => 67
        | (1, 12) => 74
        | (1, 6) => 46
        | (2, 11) => 60
        | (2, 12) => 74
        | (2, 6) => 66
        | (3, 11) => 25
        | (3, 12) => 51
        | (3, 6) => 57
        | (4, 0) => 11
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_App ) (CtorExpr_Bool )))); 
        (* 9 *) (match (size, stack1) with
        | (1, 11) => 83
        | (1, 12) => 89
        | (1, 6) => 85
        | (2, 11) => 53
        | (2, 12) => 54
        | (2, 6) => 26
        | (3, 11) => 10
        | (3, 12) => 43
        | (3, 6) => 29
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Var ) (CtorExpr_Abs )))); 
        (* 10 *) (match (size, stack1) with
        | (1, 11) => 10
        | (1, 12) => 10
        | (1, 6) => 10
        | (2, 11) => 10
        | (2, 12) => 10
        | (2, 6) => 10
        | (3, 11) => 10
        | (3, 12) => 43
        | (3, 6) => 29
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Bool ) (CtorExpr_Abs )))); 
        (* 11 *) (match (size, stack1) with
        | (1, 11) => 90
        | (1, 12) => 90
        | (1, 6) => 90
        | (2, 11) => 80
        | (2, 12) => 90
        | (2, 6) => 90
        | (3, 11) => 59
        | (3, 12) => 76
        | (3, 6) => 88
        | (4, 0) => 90
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Abs ) (CtorExpr_Abs )))); 
        (* 12 *) (match (size, stack1) with
        | (1, 11) => 74
        | (1, 12) => 84
        | (1, 6) => 85
        | (2, 11) => 85
        | (2, 12) => 90
        | (2, 6) => 90
        | (3, 11) => 90
        | (3, 12) => 51
        | (3, 6) => 55
        | (4, 0) => 15
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_App ) (CtorExpr_Abs )))); 
        (* 13 *) (match (size, stack1) with
        | (1, 11) => 56
        | (1, 12) => 64
        | (1, 6) => 64
        | (2, 11) => 62
        | (2, 12) => 75
        | (2, 6) => 60
        | (3, 11) => 13
        | (3, 12) => 43
        | (3, 6) => 29
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Var ) (CtorExpr_App )))); 
        (* 14 *) (match (size, stack1) with
        | (1, 11) => 10
        | (1, 12) => 10
        | (1, 6) => 10
        | (2, 11) => 10
        | (2, 12) => 10
        | (2, 6) => 10
        | (3, 11) => 10
        | (3, 12) => 43
        | (3, 6) => 29
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Bool ) (CtorExpr_App )))); 
        (* 15 *) (match (size, stack1) with
        | (1, 11) => 83
        | (1, 12) => 78
        | (1, 6) => 71
        | (2, 11) => 84
        | (2, 12) => 90
        | (2, 6) => 90
        | (3, 11) => 90
        | (3, 12) => 50
        | (3, 6) => 45
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Abs ) (CtorExpr_App )))); 
        (* 16 *) (match (size, stack1) with
        | (1, 11) => 44
        | (1, 12) => 22
        | (1, 6) => 32
        | (2, 11) => 87
        | (2, 12) => 90
        | (2, 6) => 90
        | (3, 11) => 90
        | (3, 12) => 45
        | (3, 6) => 34
        | (4, 0) => 10
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_App ) (CtorExpr_App ))))]) 
      (fun param_variantis => (let '(MkCtorExprCtorExpr ctor1 ctor2) := param_variantis in

        (bindGen (genExpr size1 ctor1 6) 
        (fun p1 => 
          (bindGen (genExpr size1 ctor2 12) 
          (fun p2 => 
            (returnGen (App p1 p2)))))))))
    end
  end.

Definition gSized :=

  (bindGen 
  (* Frequency1 *) (freq [
    (* 1 *) (match (tt) with
    | tt => 10
    end,
    (returnGen (CtorExpr_Var ))); 
    (* 2 *) (match (tt) with
    | tt => 10
    end,
    (returnGen (CtorExpr_Bool ))); 
    (* 3 *) (match (tt) with
    | tt => 90
    end,
    (returnGen (CtorExpr_Abs ))); 
    (* 4 *) (match (tt) with
    | tt => 10
    end,
    (returnGen (CtorExpr_App )))]) 
  (fun init_ctor => (genExpr 4 init_ctor 0))).

Definition test_prop_SinglePreserve :=
forAll gSized (fun (e: Expr) =>
  prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)

Definition test_prop_MultiPreserve :=
forAll gSized (fun (e: Expr) =>
  prop_MultiPreserve e).

(*! QuickChick test_prop_MultiPreserve. *)
          
